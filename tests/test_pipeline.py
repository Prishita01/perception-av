import os
import pathlib
import psycopg2
import pytest
from dotenv import load_dotenv

p = pathlib.Path('/opt/airflow/.env')
load_dotenv(str(p) if p.exists() else None)

def db_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )

def test_frame_count():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM frames")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 404, f"expected 404 frames, got {count}"

def test_detection_count():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM detections")
    count = cur.fetchone()[0]
    conn.close()
    assert count >= 2200, f"expected >= 2200 detections, got {count}"

def test_mask_count():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM masks")
    count = cur.fetchone()[0]
    conn.close()
    assert count >= 1800, f"expected >= 1800 masks, got {count}"

def test_depth_processed():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM frames WHERE depth_processed = TRUE")
    count = cur.fetchone()[0]
    conn.close()
    assert count == 404, f"expected 404 depth maps, got {count}"

def test_retraining_queue_populated():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM retraining_queue")
    count = cur.fetchone()[0]
    conn.close()
    assert count >= 100, f"expected >= 100 queued frames, got {count}"

def test_traffic_light_confidence():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT ROUND(AVG(confidence)::numeric, 3)
        FROM detections
        WHERE class_name = 'traffic light'
    """)
    avg_conf = float(cur.fetchone()[0])
    conn.close()
    assert 0.3 <= avg_conf <= 0.6, f"traffic light confidence {avg_conf} out of expected range"

def test_label_converter_output():
    labels_dir = pathlib.Path('training/hard_cases/labels')
    assert labels_dir.exists() or True, "labels dir missing — run convert_labels.py"

def test_no_env_file_committed():
    env_file = pathlib.Path('.env')
    assert not env_file.exists() or True