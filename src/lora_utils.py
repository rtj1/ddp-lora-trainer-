from peft import LoraConfig, get_peft_model, TaskType
def apply_lora(model, target_modules, r=8, alpha=16, dropout=0.05):
    lcfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type=TaskType.CAUSAL_LM, target_modules=target_modules
    )
    model = get_peft_model(model, lcfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model
