FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn9-runtime
WORKDIR /workspace
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
CMD ["bash", "-lc", "python -m src.train --config configs/base.yaml training.max_steps=50"]
