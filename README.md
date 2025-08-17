# 🚀 Distributed LoRA Trainer  
**Reproducible Infrastructure for Fine-Tuning Large Language Models**  
*DDP · FSDP · DeepSpeed · QLoRA · W&B · Airflow · Kubernetes*  

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)  
[![Transformers](https://img.shields.io/badge/huggingface-transformers-yellow.svg)](https://huggingface.co/transformers/)  
[![DeepSpeed](https://img.shields.io/badge/deepspeed-enabled-green.svg)](https://github.com/microsoft/DeepSpeed)  
[![W&B](https://img.shields.io/badge/Weights_&_Biases-logging-orange.svg)](https://wandb.ai/)  

---

## 📌 Overview
This project provides a **production-grade, reproducible training infrastructure** for fine-tuning LLMs with parameter-efficient adapters (LoRA/QLoRA). It’s designed for:  
- **Scalable distributed training** across multi-GPU/multi-node clusters.  
- **Experiment reproducibility** with config-driven workflows.  
- **High throughput** with sharded streaming datasets + optimized I/O.  
- **Research enablement**: rapid prototyping of new training methods.  


---

## ✨ Features
- ⚡ **Distributed Backends:** PyTorch DDP, FSDP, DeepSpeed (Zero-2/3).  
- 💾 **Parameter-Efficient Fine-Tuning:** LoRA + 4-bit QLoRA (bitsandbytes).  
- 📊 **Experiment Tracking:** Automated logging, checkpointing, and sweeps with **Weights & Biases**.  
- 🗂 **Streaming Datasets:** Shard-aware Hugging Face `datasets` with per-rank partitioning.  
- 🧩 **Config-Driven:** YAML + `.env` toggles for reproducibility.  
- ☸️ **Orchestration:** Airflow DAGs + Kubernetes CronJobs for scheduled training.  
- 🛠 **Resilience:** Standardized checkpoints, monitoring, and failure recovery.  

---

## 📂 Project Structure
```
ddp-lora-trainer/
│
├── configs/                # YAML configs
├── src/                    # Core training code
├── scripts/                # Launchers & sweep runners
├── sweeps/                 # W&B sweep configs
├── orchestration/
│   ├── airflow/            # Airflow DAGs
│   └── k8s/                # Kubernetes manifests
├── deepspeed/              # ZeRO configs
├── Dockerfile
├── README.md
└── .env.example
```

---

## 🚀 Quickstart

### 1. Install
```bash
git clone https://github.com/<your-username>/ddp-lora-trainer.git
cd ddp-lora-trainer
pip install -r requirements.txt
```

### 2. Local DDP (4 GPUs)
```bash
bash scripts/launch_local.sh 4 configs/base.yaml
```

### 3. Hugging Face Streaming Dataset
```bash
python -m src.train --config configs/base.yaml   data.use_hf_streaming=true data.hf_dataset=wikitext data.hf_split=train
```

### 4. DeepSpeed
```bash
python -m src.train --config configs/base.yaml   distributed.backend=deepspeed distributed.deepspeed_config=./deepspeed/zero2.json
```

### 5. FSDP
```bash
python -m src.train --config configs/base.yaml   distributed.backend=fsdp fsdp.sharding_strategy=full fsdp.mixed_precision=bf16
```

### 6. QLoRA (4-bit)
```bash
python -m src.train --config configs/base.yaml   quantization.load_in_4bit=true quantization.compute_dtype=bf16
```

---

## 📊 Weights & Biases
Run training with W&B logging:
```bash
WANDB_PROJECT=ddp-lora-trainer WANDB_API_KEY=your_key bash scripts/launch_local.sh 4 configs/base.yaml
```
Run a sweep:
```bash
bash scripts/wandb_sweep.sh
```

---

## ☸️ Orchestration

### Airflow DAG
- Located in `orchestration/airflow/dags/train_ddp_lora.py`.  
- Configure via Airflow Variables (`ddp_lora_repo_path`, `hf_dataset`, etc.).  

### Kubernetes CronJob
- `orchestration/k8s/cronjob.yaml` schedules recurring training jobs.  
- Requires GPU node pool + `wandb-secret` + PVC `ddp-lora-pvc`.  

---

## 📈 Benchmarking
```bash
python scripts/bench_efficiency.py --config configs/base.yaml --gpus 8
```
Generates scaling efficiency plots (tokens/sec vs GPUs).  

---

## 📝 License
Apache-2.0  

---

⚡ Maintainer: [Your Name](https://github.com/<your-username>)  
