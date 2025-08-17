import argparse, os, time
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers import BitsAndBytesConfig
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
import wandb

import contextlib
try:
    import deepspeed
except Exception:
    deepspeed = None

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType
    from torch.distributed.fsdp import FullStateDictConfig, LocalStateDictConfig
    from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
except Exception:
    FSDP = None
    ShardingStrategy = None
    MixedPrecision = None
    StateDictType = None
    FullStateDictConfig = None
    LocalStateDictConfig = None

def set_sdpa_policy(mode: str):
    """Set PyTorch SDP/FlashAttention policy if available."""
    try:
        if mode == "flash":
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        elif mode == "mem_efficient":
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
        else:
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        pass


from .utils import load_config, override_from_kv, set_seed, ddp_info, is_main_process, barrier, ThroughputMeter
from .dataset import build_dataloader
from .lora_utils import apply_lora

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local if torch.cuda.is_available() else 0)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        return rank, world, local
    return 0,1,0

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("overrides", nargs="*")
    return ap.parse_args()

def maybe_wandb(cfg):
    if cfg["logging"]["wandb"].get("enabled", True) and not os.environ.get("WANDB_DISABLED") and is_main_process():
        wandb.init(
            project=cfg["logging"]["wandb"].get("project","ddp-lora-llm"),
            entity=cfg["logging"]["wandb"].get("entity") or None,
            name=cfg["logging"]["wandb"].get("run_name"),
            config=cfg, resume="allow",
        ); return True
    return False

def save_checkpoint(path, model, optimizer, scheduler, scaler, step, cfg, tokenizer, lora_only=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "scaler": scaler.state_dict() if scaler is not None else None, "config": cfg}
    torch.save(state, path)
    if hasattr(model, "module"): m = model.module
    else: m = model
    if lora_only and hasattr(m, "peft_config"):
        adapter_dir = os.path.join(os.path.dirname(path), "adapter")
        m.save_pretrained(adapter_dir); tokenizer.save_pretrained(adapter_dir)
    else:
        full_dir = os.path.join(os.path.dirname(path), "full")
        m.save_pretrained(full_dir); tokenizer.save_pretrained(full_dir)

def load_resume(path, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"])
    if ckpt.get("scaler") and scaler is not None: scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("step", 0)

def main():
    args = parse_args()
    cfg = load_config(args.config); cfg = override_from_kv(cfg, args.overrides)
    set_seed(cfg.get("seed",42))

    rank, world, local = init_distributed()
    device = "cuda" if (torch.cuda.is_available() and cfg["device"].get("cuda_if_available",True)) else "cpu"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer_name"])
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(cfg["model"].get("dtype","bf16"), torch.bfloat16)

    # SDPA/Flash policy
    set_sdpa_policy(cfg.get("attention", {}).get("sdpa", "flash"))

    # BitsAndBytes 4-bit (QLoRA) optional
    quant = cfg.get("quantization", {})
    bnb_config = None
    model_kwargs = {}
    if quant.get("load_in_4bit", False):
        compute_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[quant.get("compute_dtype","bf16")]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type","nf4"),
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs.update(dict(device_map="auto", quantization_config=bnb_config))
    else:
        model_kwargs.update(dict(torch_dtype=dtype if device=="cuda" else None))

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], **model_kwargs)

    if cfg["lora"].get("enabled", True):
        model = apply_lora(model, target_modules=cfg["lora"].get("target_modules",[]), r=cfg["lora"].get("r",8),
                           alpha=cfg["lora"].get("alpha",16), dropout=cfg["lora"].get("dropout",0.05))

    backend = cfg.get("distributed", {}).get("backend", "ddp")
    model.to(device)

    fsdp_model = None
    ds_engine = None
    model_ref = model

    if backend == "fsdp":
        assert FSDP is not None, "FSDP not available in this environment"
        mp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg.get("fsdp",{}).get("mixed_precision","bf16")]
        mp = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        strat_map = {"full": ShardingStrategy.FULL_SHARD, "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP, "hybrid": ShardingStrategy.HYBRID_SHARD}
        sharding = strat_map.get(cfg.get("fsdp",{}).get("sharding_strategy","full"), ShardingStrategy.FULL_SHARD)
        model_ref = FSDP(model, mixed_precision=mp, sharding_strategy=sharding, use_orig_params=True, device_id=(local if device=='cuda' else None))

    if backend == "ddp":
        if world>1 and device=="cuda":
            model_ref = DDP(model_ref, device_ids=[local], output_device=local, find_unused_parameters=False)
        elif world>1:
            model_ref = DDP(model_ref)  # CPU fallback

    loader, collate = build_dataloader(tok, cfg["data"], cfg["training"]["batch_size_per_device"], rank, world)

    optim = AdamW((p for p in model_ref.parameters() if p.requires_grad),
                  lr=cfg["training"]["lr"], betas=(cfg["training"]["adam_beta1"], cfg["training"]["adam_beta2"]),
                  eps=cfg["training"]["adam_eps"], weight_decay=cfg["training"]["weight_decay"])
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=cfg["training"]["warmup_steps"], num_training_steps=cfg["training"]["max_steps"])

    if backend == "deepspeed":
        assert deepspeed is not None, "Install deepspeed to use distributed.backend=deepspeed"
        import json as _json
        ds_cfg_path = cfg.get("distributed", {}).get("deepspeed_config", "./deepspeed/zero2.json")
        with open(ds_cfg_path, "r") as _f:
            ds_cfg = _json.load(_f)
        params = [p for p in model_ref.parameters() if p.requires_grad]
        model_ref, optim, _, sched = deepspeed.initialize(model=model_ref, model_parameters=params, config=ds_cfg, optimizer=optim, lr_scheduler=sched)

    scaler = GradScaler(enabled=(device=="cuda" and cfg["model"].get("dtype")=="fp16" and backend!='deepspeed'))

    step = 0
    if cfg["checkpoint"].get("resume_from"):
        if is_main_process(): print("Resuming from:", cfg["checkpoint"]["resume_from"])
        step = load_resume(cfg["checkpoint"]["resume_from"], optim, sched, scaler)

    wb = maybe_wandb(cfg)

    model.train()
    loss_acc = 0.0
    tpt = ThroughputMeter()
    t0 = time.time()

    for epoch in range(10**9):
        for batch in loader:
            batch = collate(batch)
            x = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            use_amp = (device=="cuda" and cfg["model"].get("dtype") in ["bf16","fp16"])
            amp_dtype = torch.bfloat16 if cfg["model"].get("dtype")=="bf16" else torch.float16

            with autocast(enabled=use_amp, dtype=amp_dtype):
                out = model_ref(input_ids=x, attention_mask=attn, labels=y)
                loss = out.loss / cfg["training"]["grad_accum_steps"]

            if cfg.get('distributed',{}).get('backend','ddp') == 'deepspeed':
                model_ref.backward(loss)
            else:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            loss_acc += loss.item()

            if (step+1) % cfg["training"]["grad_accum_steps"] == 0:
                if cfg["training"]["max_grad_norm"]:
                    if scaler.is_enabled(): scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["max_grad_norm"])

                if cfg.get('distributed',{}).get('backend','ddp') == 'deepspeed':
                    model_ref.step()
                else:
                    if scaler.is_enabled():
                        scaler.step(optim); scaler.update()
                    else:
                        optim.step()
                    optim.zero_grad(set_to_none=True); sched.step()

            if is_main_process() and (step % cfg["training"]["log_interval"] == 0):
                dt = max(time.time()-t0, 1e-8)
                toks = int(attn.sum().item())
                tpt.update(tokens=toks*(world if dist.is_initialized() else 1), dt=dt)
                t0 = time.time()
                log = {"step": step, "loss": loss.item()*cfg["training"]["grad_accum_steps"], "lr": sched.get_last_lr()[0], "tokens/sec": tpt.avg_toks_per_sec()}
                print(f"[step {step}] loss={log['loss']:.4f} lr={log['lr']:.6g} toks/sec~{log['tokens/sec']:.1f}")
                if wb: wandb.log(log, step=step)

            if is_main_process() and cfg["training"]["save_interval"] and (step % cfg["training"]["save_interval"]==0) and step>0:
                save_checkpoint(os.path.join(cfg["checkpoint"]["output_dir"], "last.pt"), model, optim, sched, scaler, step, cfg, tok, cfg["checkpoint"].get("save_lora_only", True))

            step += 1
            if step >= cfg["training"]["max_steps"]:
                break
        if step >= cfg["training"]["max_steps"]:
            break

    if is_main_process():
        print("Training complete.")
        save_checkpoint(os.path.join(cfg["checkpoint"]["output_dir"], "last.pt"), model, optim, sched, scaler, step, cfg, tok, cfg["checkpoint"].get("save_lora_only", True))

    if dist.is_initialized():
        barrier(); dist.destroy_process_group()

if __name__ == "__main__":
    main()
