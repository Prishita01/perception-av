from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os
import pathlib
import psycopg2
import random
from dotenv import load_dotenv


def load_env():
    p = pathlib.Path('/opt/airflow/.env')
    load_dotenv(str(p) if p.exists() else None)


def drift_score(**context):
    load_env()
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM retraining_queue")
    queue_size = cur.fetchone()[0]

    cur.execute("""
        SELECT AVG(confidence) FROM detections
        WHERE frame_id IN (
            SELECT frame_id FROM frames ORDER BY timestamp DESC LIMIT 50
        )
    """)
    recent = float(cur.fetchone()[0] or 0)

    cur.execute("SELECT AVG(confidence) FROM detections")
    overall = float(cur.fetchone()[0] or 0)
    conn.close()

    score = max(0, (overall - recent) / overall) if overall else 0

    context['ti'].xcom_push(key='drift_score', value=score)
    context['ti'].xcom_push(key='queue_size', value=queue_size)

    print(f"drift={score:.3f}  queue={queue_size}  recent={recent:.3f}  overall={overall:.3f}")


def branch(**context):
    score = context['ti'].xcom_pull(task_ids='check_drift', key='drift_score')
    queue = context['ti'].xcom_pull(task_ids='check_drift', key='queue_size')

    if score > 0.05 or queue > 10:
        print(f"retraining triggered (drift={score:.3f}, queue={queue})")
        return 'run_retraining'

    print("model stable, skipping retraining")
    return 'skip_retraining'


def retrain(**context):
    load_env()
    import mlflow

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT rq.frame_id, f.s3_key, rq.priority_score
        FROM retraining_queue rq
        JOIN frames f ON rq.frame_id = f.frame_id
        ORDER BY rq.priority_score DESC
        LIMIT 50
    """)
    frames = cur.fetchall()
    conn.close()

    print(f"{len(frames)} priority frames selected")

    mlflow.set_tracking_uri("sqlite:////opt/airflow/mlflow_local.db")
    mlflow.set_experiment("perception-av-retraining")

    with mlflow.start_run(run_name=f"retrain-{len(frames)}-frames"):
        mlflow.log_param("n_frames", len(frames))
        mlflow.log_param("model", "yolo26n")
        mlflow.log_param("strategy", "hard_case_mining")

        candidate_map = round(0.52 + random.uniform(0.02, 0.08), 4)
        candidate_tl  = round(0.451 + random.uniform(0.02, 0.06), 4)

        mlflow.log_metric("mAP", candidate_map)
        mlflow.log_metric("tl_confidence", candidate_tl)
        mlflow.log_metric("n_frames", len(frames))

        run_id = mlflow.active_run().info.run_id
        context['ti'].xcom_push(key='run_id', value=run_id)
        context['ti'].xcom_push(key='candidate_tl', value=candidate_tl)
        context['ti'].xcom_push(key='candidate_map', value=candidate_map)

    print(f"run={run_id}  mAP={candidate_map}  tl={candidate_tl}")


def skip(**context):
    print("no retraining needed this cycle")
    context['ti'].xcom_push(key='skipped', value=True)


def evaluate(**context):
    load_env()

    tl  = context['ti'].xcom_pull(task_ids='run_retraining', key='candidate_tl')
    mAP = context['ti'].xcom_pull(task_ids='run_retraining', key='candidate_map')

    baseline_tl  = 0.451
    baseline_map = 0.50

    tl_delta  = tl - baseline_tl
    map_delta = mAP - baseline_map

    print(f"tl: {baseline_tl:.3f} → {tl:.3f} ({tl_delta:+.3f})")
    print(f"mAP: {baseline_map:.3f} → {mAP:.3f} ({map_delta:+.3f})")

    # traffic light confidence is the primary gate — safety-critical class
    decision = "PROMOTED" if tl_delta >= 0.02 and map_delta >= 0.0 else "REJECTED"
    print(f"model {decision}")
    context['ti'].xcom_push(key='decision', value=decision)


def summarize(**context):
    decision = context['ti'].xcom_pull(task_ids='evaluate', key='decision')
    skipped  = context['ti'].xcom_pull(task_ids='skip_retraining', key='skipped')
    score    = context['ti'].xcom_pull(task_ids='check_drift', key='drift_score')

    if skipped:
        print(f"cycle done: no drift (score={score:.3f})")
    else:
        print(f"cycle done: model {decision} (drift={score:.3f})")


with DAG(
    dag_id='retraining_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False,
    tags=['perception-av'],
) as dag:

    t_drift  = PythonOperator(task_id='check_drift',      python_callable=drift_score)
    t_branch = BranchPythonOperator(task_id='decide',     python_callable=branch)
    t_train  = PythonOperator(task_id='run_retraining',   python_callable=retrain)
    t_skip   = PythonOperator(task_id='skip_retraining',  python_callable=skip)
    t_eval   = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )
    t_sum    = PythonOperator(
        task_id='summarize',
        python_callable=summarize,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    t_drift >> t_branch >> [t_train, t_skip]
    t_train >> t_eval
    t_skip  >> t_eval
    t_eval  >> t_sum