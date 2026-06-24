from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from dotenv import load_dotenv
import os
import boto3
import psycopg2


def check_rds(**context):
    load_dotenv('/opt/airflow/.env')
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM frames')
    count = cur.fetchone()[0]
    conn.close()
    print(f"rds ok — {count} frames")
    context['ti'].xcom_push(key='frame_count', value=count)


def check_s3(**context):
    load_dotenv('/opt/airflow/.env')
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET_NAME')
    resp = s3.list_objects_v2(Bucket=bucket, Prefix='frames/')
    count = resp.get('KeyCount', 0)
    print(f"s3 ok — {count} objects")
    context['ti'].xcom_push(key='s3_count', value=count)


def verify(**context):
    load_dotenv('/opt/airflow/.env')

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    cur = conn.cursor()

    cur.execute('SELECT COUNT(*) FROM frames')
    frames = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM detections')
    dets = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM frames WHERE depth_processed = TRUE')
    depth = cur.fetchone()[0]

    cur.execute('SELECT COUNT(*) FROM masks')
    masks = cur.fetchone()[0]

    conn.close()

    if frames != 404:
        raise Exception(f"frame count off: {frames}")
    if dets < 2000:
        raise Exception(f"detections too low: {dets}")
    if depth < 400:
        raise Exception(f"depth maps too low: {depth}")
    if masks < 1500:
        raise Exception(f"masks too low: {masks}")

    summary = f"frames={frames} detections={dets} depth={depth} masks={masks}"
    context['ti'].xcom_push(key='summary', value=summary)
    print(summary)


def report(**context):
    summary = context['ti'].xcom_pull(task_ids='verify', key='summary')
    print(f"pipeline complete: {summary}")


with DAG(
    dag_id='ingestion_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule='@daily',
    catchup=False,
    tags=['perception-av'],
) as dag:

    t_rds = PythonOperator(task_id='check_rds_connection', python_callable=check_rds)
    t_s3  = PythonOperator(task_id='check_s3_connection',  python_callable=check_s3)

    t_ingest = BashOperator(
        task_id='run_ingestion',
        bash_command='cd /opt/airflow && python ingestion/ingest.py',
    )
    t_detect = BashOperator(
        task_id='run_detection',
        bash_command='curl -sf -m 3600 -X POST http://ml-worker:8000/detect',
    )
    t_depth = BashOperator(
        task_id='run_depth',
        bash_command='curl -sf -m 3600 -X POST http://ml-worker:8000/depth',
    )
    t_segment = BashOperator(
        task_id='run_sam2',
        bash_command='curl -sf -m 3600 -X POST http://ml-worker:8000/segment',
    )
    t_verify = PythonOperator(task_id='verify', python_callable=verify)
    t_report = PythonOperator(task_id='report_results', python_callable=report)

    [t_rds, t_s3] >> t_ingest >> t_detect >> t_depth >> t_segment >> t_verify >> t_report