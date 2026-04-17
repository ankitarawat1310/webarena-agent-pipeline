import os
import sys
import json
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

DATA_DIR = Path("/app/data")
SCRIPTS_DIR = Path("/app/scripts")
DBFS_ROOT = "/webarena"

DEFAULT_ARGS = {
    "owner": "team4-data298a",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
    "email_on_failure": False,
}


def _run_in_collector(script, args=None):
    cmd = ["docker", "exec", "-w", "/app", "webarena-collector", "python", f"/app/scripts/{script}"]
    if args:
        cmd += args
    log.info(f"Running in collector: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout:
        log.info(r.stdout)
    if r.stderr:
        log.warning(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stderr}")
    return r.stdout


def _run_local(script, args=None):
    cmd = [sys.executable, f"/opt/airflow/scripts/{script}"]
    if args:
        cmd += args
    log.info(f"Running locally: {' '.join(cmd)}")
    r = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd="/opt/airflow",
        env={**os.environ, "PYTHONPATH": "/opt/airflow"}
    )
    if r.stdout:
        log.info(r.stdout)
    if r.stderr:
        log.warning(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stderr}")
    return r.stdout


def collect(**context):
    run_id = context["ds_nodash"]
    context["ti"].xcom_push(key="run_id", value=run_id)
    log.info("Starting data collection in collector container...")
    _run_in_collector("collect_data.py")
    log.info("Collection complete")


def export_raw(**context):
    _run_in_collector("export_raw_dataset.py")
    log.info("Raw dataset exported")


def preprocess(**context):
    _run_in_collector("preprocess_data.py")
    log.info("Preprocessing complete")


def transform(**context):
    _run_in_collector("transform_data.py")
    log.info("Transform complete")


def split(**context):
    _run_in_collector("split_data.py")
    splits = {}
    for s in ["train", "val", "test"]:
        r = subprocess.run(
            ["docker", "exec", "webarena-collector", "wc", "-l", f"/app/data/splits/{s}.jsonl"],
            capture_output=True, text=True
        )
        count = int(r.stdout.strip().split()[0]) if r.returncode == 0 else 0
        splits[s] = count
    context["ti"].xcom_push(key="split_counts", value=splits)
    log.info(f"Splits: {splits}")
    return splits


def upload_databricks(**context):
    run_id = context["ti"].xcom_pull(task_ids="collect_data", key="run_id") or context["ds_nodash"]
    _run_local("databricks_uploader.py", [
        "--run-id", run_id,
        "--data-root", "/opt/airflow/data",
        "--volume-root", "/Volumes/workspace/webarena/trajectories",
    ])
    return run_id


def pipeline_summary(**context):
    ti = context["ti"]
    summary = {
        "run_id": ti.xcom_pull(task_ids="collect_data", key="run_id"),
        "splits": ti.xcom_pull(task_ids="split_data", key="split_counts") or {},
        "databricks_path": f"{DBFS_ROOT}/{ti.xcom_pull(task_ids='collect_data', key='run_id')}/",
    }
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE — TEAM 4 DATA 298A")
    log.info(json.dumps(summary, indent=2))
    log.info("=" * 60)
    return summary


with DAG(
    dag_id="webarena_verified_pipeline",
    default_args=DEFAULT_ARGS,
    description="Team 4: WebArena trajectory collection to Databricks",
    schedule_interval="0 2 * * *",
    start_date=datetime(2026, 4, 1),
    catchup=False,
    tags=["webarena", "team4", "data298a"],
    max_active_runs=1,
) as dag:

    t_collect = PythonOperator(task_id="collect_data",         python_callable=collect)
    t_export  = PythonOperator(task_id="export_raw_dataset",   python_callable=export_raw)
    t_pre     = PythonOperator(task_id="preprocess_data",      python_callable=preprocess)
    t_trans   = PythonOperator(task_id="transform_data",       python_callable=transform)
    t_split   = PythonOperator(task_id="split_data",           python_callable=split)
    t_upload  = PythonOperator(task_id="upload_to_databricks", python_callable=upload_databricks)
    t_summary = PythonOperator(task_id="pipeline_summary",     python_callable=pipeline_summary,
                               trigger_rule=TriggerRule.ALL_DONE)

    t_collect >> t_export >> t_pre >> t_trans >> t_split >> t_upload >> t_summary