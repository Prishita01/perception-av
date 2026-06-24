from transformers import pipeline
from PIL import Image
import numpy as np
import boto3
import psycopg2
import os
import io
import pathlib
import tempfile
import torch
from dotenv import load_dotenv


def load_env():
    p = pathlib.Path('/opt/airflow/.env')
    load_dotenv(str(p) if p.exists() else None)


def pending_frames(conn):
    cur = conn.cursor()
    cur.execute(
        "SELECT frame_id, s3_key FROM frames WHERE depth_processed = FALSE"
    )
    return cur.fetchall()


def upload_depth(s3, bucket, arr, frame_id):
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    key = f"depth_maps/{frame_id}.npy"
    s3.upload_fileobj(buf, bucket, key)
    return key


def mark_done(conn, frame_id, s3_key):
    cur = conn.cursor()
    cur.execute(
        "UPDATE frames SET depth_s3_key = %s, depth_processed = TRUE WHERE frame_id = %s",
        (s3_key, frame_id)
    )
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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )

    frames = pending_frames(conn)
    print(f"{len(frames)} frames queued")

    ok, err = 0, 0
    for frame_id, s3_key in frames:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                s3.download_fileobj(bucket, s3_key, tmp)
                path = tmp.name

            depth = np.array(pipe(Image.open(path))["depth"])
            key = upload_depth(s3, bucket, depth, frame_id)
            mark_done(conn, frame_id, key)
            os.unlink(path)

            ok += 1
            if ok % 50 == 0:
                print(f"  {ok}/{len(frames)}")

        except Exception as e:
            print(f"  {frame_id[:8]} failed: {e}")
            err += 1

    print(f"\n{ok} processed, {err} errors")
    conn.close()


if __name__ == '__main__':
    main()