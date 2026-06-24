from ultralytics import YOLO
from dotenv import load_dotenv
import boto3
import psycopg2
import os
import pathlib
import tempfile


def load_env():
    p = pathlib.Path('/opt/airflow/.env')
    load_dotenv(str(p) if p.exists() else None)


def pending_frames(conn):
    cur = conn.cursor()
    cur.execute(" SELECT f.frame_id, f.s3_key FROM frames f LEFT JOIN detections d ON f.frame_id = d.frame_id  WHERE d.frame_id IS NULL")
    return cur.fetchall()


def run_detection(model, image_path):
    results = model(image_path, verbose=False)
    detections = []
    for r in results:
        for i in range(len(r.boxes)):
            detections.append({
                'class_id':   int(r.boxes.cls[i].item()),
                'class_name': model.names[int(r.boxes.cls[i].item())],
                'confidence': float(r.boxes.conf[i].item()),
                'x1': float(r.boxes.xyxy[i][0].item()),
                'y1': float(r.boxes.xyxy[i][1].item()),
                'x2': float(r.boxes.xyxy[i][2].item()),
                'y2': float(r.boxes.xyxy[i][3].item()),
            })
    return detections


def insert_detections(conn, frame_id, detections):
    cur = conn.cursor()
    for d in detections:
        cur.execute("""
            INSERT INTO detections
                (frame_id, class_id, class_name, confidence,
                 bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            frame_id, d['class_id'], d['class_name'], d['confidence'],
            d['x1'], d['y1'], d['x2'], d['y2'],
        ))
    conn.commit()
    cur.close()


def main():
    load_env()

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    s3 = boto3.client('s3')
    bucket = os.getenv('S3_BUCKET_NAME')

    model = YOLO("yolo26n.pt")
    frames = pending_frames(conn)
    print(f"{len(frames)} frames queued")

    ok, err = 0, 0
    for frame_id, s3_key in frames:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                s3.download_fileobj(bucket, s3_key, tmp)
                path = tmp.name

            dets = run_detection(model, path)
            insert_detections(conn, frame_id, dets)
            os.unlink(path)

            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{len(frames)}")

        except Exception as e:
            print(f"  {frame_id[:8]} failed: {e}")
            err += 1

    print(f"\n{ok} processed, {err} errors")

    cur = conn.cursor()
    cur.execute("""
        SELECT class_name, COUNT(*) as n,
               ROUND(AVG(confidence)::numeric, 3) as avg_conf
        FROM detections
        GROUP BY class_name
        ORDER BY n DESC
    """)
    for row in cur.fetchall():
        print(f"  {row[0]:<16} {row[1]:>5}  conf={row[2]}")

    conn.close()


if __name__ == '__main__':
    main()