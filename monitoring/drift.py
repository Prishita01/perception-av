import os
import pathlib
import tempfile
from datetime import datetime

import boto3
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from evidently import Report
from evidently.presets import DataDriftPreset

p = pathlib.Path('/opt/airflow/.env')
load_dotenv(str(p) if p.exists() else None)


def confidence_by_frame(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT
            d.frame_id,
            f.timestamp,
            AVG(CASE WHEN d.class_name = 'car' THEN d.confidence END) as car_conf,
            AVG(CASE WHEN d.class_name = 'person' THEN d.confidence END) as person_conf,
            AVG(CASE WHEN d.class_name = 'traffic light' THEN d.confidence END) as tl_conf,
            COUNT(*) as total_detections
        FROM detections d
        JOIN frames f ON d.frame_id = f.frame_id
        GROUP BY d.frame_id, f.timestamp
        ORDER BY f.timestamp ASC
    """)
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=[
        'frame_id', 'timestamp', 'car_conf',
        'person_conf', 'tl_conf', 'total_detections'
    ])


def check_drift(df):
    df = df.fillna(0)
    mid = len(df) // 2
    features = ['car_conf', 'person_conf', 'tl_conf', 'total_detections']

    ref = df.iloc[:mid][features].reset_index(drop=True)
    cur = df.iloc[mid:][features].reset_index(drop=True)

    report = Report([DataDriftPreset()])
    result = report.run(cur, ref)

    snapshot = result.dict()
    drift_share = 0.0
    drifted = False

    for metric in snapshot.get('metrics', []):
        name = metric.get('metric_name', '')
        value = metric.get('value', {})
        if 'DriftedColumnsCount' in name:
            drift_share = float(value.get('share', 0.0))
            drifted = drift_share > 0.3
            break
        
        if isinstance(value, dict):
            if 'share_of_drifted_columns' in value:
                drift_share = float(value['share_of_drifted_columns'])
                drifted = drift_share > 0.3
                break
            if 'drift_share' in value:
                drift_share = float(value['drift_share'])
                drifted = drift_share > 0.3
                break

    return report, result, drift_share, drifted


def save_report(result, s3, bucket):
    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
        path = tmp.name
    result.save_html(path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    key = f"drift_reports/report_{timestamp}.html"
    s3.upload_file(path, bucket, key)
    os.unlink(path)
    return key


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET_NAME')

    df = confidence_by_frame(conn)
    print(f"{len(df)} frames loaded")

    report, result, drift_score, drifted = check_drift(df)
    print(f"drift score: {drift_score:.3f}  {'DRIFT DETECTED' if drifted else 'No drift'}")

    try:
        key = save_report(result, s3, bucket)
        print(f"report → s3://{bucket}/{key}")
    except Exception as e:
        print(f"report upload failed: {e}")

    conn.close()
    return drift_score


if __name__ == '__main__':
    main()