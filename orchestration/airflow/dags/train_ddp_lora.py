from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="ddp_lora_train_daily",
    default_args=default_args,
    description="Daily DDP LoRA training run",
    schedule_interval="0 2 * * *",
    start_date=datetime(2025, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["llm","ddp","lora"],
) as dag:
    train = BashOperator(
        task_id="train",
        bash_command=(
            "cd {{ var.value.ddp_lora_repo_path }} && "
            "source .venv/bin/activate && "
            "WANDB_DISABLED={{ var.value.wandb_disabled | default('true') }} "
            "bash scripts/launch_local.sh 4 configs/base.yaml "
            "data.use_hf_streaming={{ var.value.use_hf_streaming | default('false') }} "
            "data.hf_dataset={{ var.value.hf_dataset | default('') }} "
            "RUN_NAME=airflow-{{ ds_nodash }}"
        ),
        env={"HF_TOKEN": "{{ var.value.hf_token | default('') }}"},
    )
