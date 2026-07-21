"""
Painting in a Painting — Airflow DAG

Full pipeline: data engineering → synthetic generation → training →
evaluation → deployment. Triggered manually (not scheduled).
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "nikhil",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="painting_pipeline",
    default_args=default_args,
    description="End-to-end pipeline for hidden painting detection",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["ml", "computer-vision", "painting"],
) as dag:

    # ──────────────────────────────────────────────
    # Phase 1 — Data Engineering (PySpark)
    # ──────────────────────────────────────────────

    bronze_ingest = BashOperator(
        task_id="bronze_ingest",
        bash_command="cd /opt/airflow && python scripts/bronze_ingest.py",
    )

    bronze_to_silver = BashOperator(
        task_id="bronze_to_silver",
        bash_command="cd /opt/airflow && python scripts/bronze_to_silver.py",
    )

    silver_to_gold = BashOperator(
        task_id="silver_to_gold",
        bash_command="cd /opt/airflow && python scripts/silver_to_gold.py",
    )

    # ──────────────────────────────────────────────
    # Phase 1.5 — Upload Gold to Azure Data Lake
    # ──────────────────────────────────────────────

    upload_gold = BashOperator(
        task_id="upload_gold_to_datalake",
        bash_command="cd /opt/airflow && python scripts/upload_gold_to_datalake.py",
    )

    # ──────────────────────────────────────────────
    # Phase 2 — Synthetic Data Generation
    # ──────────────────────────────────────────────

    generate_dataset = BashOperator(
        task_id="generate_dataset",
        bash_command="cd /opt/airflow && python -m scripts.generate_dataset",
    )

    # ──────────────────────────────────────────────
    # Phase 3 — Training (ViT + Optuna + MLflow)
    # ──────────────────────────────────────────────

    train = BashOperator(
        task_id="train",
        bash_command="cd /opt/airflow && python -m scripts.train",
        # training can take hours — extend timeout
        execution_timeout=timedelta(hours=12),
    )

    # ──────────────────────────────────────────────
    # Phase 4 — Evaluation (metrics + Grad-CAM)
    # ──────────────────────────────────────────────

    evaluate = BashOperator(
        task_id="evaluate",
        bash_command="cd /opt/airflow && python scripts/evaluate.py",
    )

    # ──────────────────────────────────────────────
    # Phase 5 — Upload checkpoint to Data Lake
    # ──────────────────────────────────────────────

    upload_checkpoint = BashOperator(
        task_id="upload_checkpoint",
        bash_command="cd /opt/airflow && python scripts/upload_checkpoint.py",
    )

    # ──────────────────────────────────────────────
    # Phase 6 — Deploy to Azure Container Apps
    # ──────────────────────────────────────────────

    deploy = BashOperator(
        task_id="deploy",
        bash_command="cd /opt/airflow && python scripts/deploy.py",
    )

    # ──────────────────────────────────────────────
    # Task dependencies
    # ──────────────────────────────────────────────

    (
        bronze_ingest
        >> bronze_to_silver
        >> silver_to_gold
        >> upload_gold
        >> generate_dataset
        >> train
        >> evaluate
        >> upload_checkpoint
        >> deploy
    )