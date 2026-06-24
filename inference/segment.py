from sam2.sam2_image_predictor import SAM2ImagePredictor
from pycocotools import mask as mask_utils
from PIL import Image
import numpy as np
import psycopg2
import boto3
import torch
import json
import os
import pathlib
import tempfile
from dotenv import load_dotenv


def load_env():
    p = pathlib.Path('/opt/airflow/.env')
    load_dotenv(str(p) if p.exists() else None)


def pending_frames(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT d.frame_id, f.s3_key
        FROM detections d
        JOIN frames f ON d.frame_id = f.frame_id
        WHERE d.frame_id NOT IN (SELECT DISTINCT frame_id FROM masks)
    """)
    return cur.fetchall()


def frame_detections(conn, frame_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, class_name, confidence,
               bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
        FROM detections
        WHERE frame_id = %s AND confidence > 0.3
    """, (frame_id,))
    return cur.fetchall()


def to_rle(mask):
    # RLE keeps mask storage ~700x smaller than raw binary
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded['counts'] = encoded['counts'].decode('utf-8')
    return json.dumps(encoded)


def insert_mask(conn, frame_id, det_id, class_name, rle, area):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO masks (frame_id, detection_id, class_name, rle_mask, area)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (frame_id, det_id, class_name, rle, area))
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
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2-hiera-tiny", device=device
    )

    frames = pending_frames(conn)
    print(f"{len(frames)} frames queued")

    ok = 0
    for frame_id, s3_key in frames:
        try:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                s3.download_fileobj(bucket, s3_key, tmp)
                path = tmp.name

            predictor.set_image(np.array(Image.open(path)))
            dets = frame_detections(conn, frame_id)

            for det_id, class_name, _, x1, y1, x2, y2 in dets:
                masks, _, _ = predictor.predict(
                    box=np.array([x1, y1, x2, y2]),
                    multimask_output=False
                )
                mask = masks[0]
                insert_mask(conn, frame_id, det_id, class_name,
                            to_rle(mask), int(mask.sum()))

            os.unlink(path)
            ok += 1
            if ok % 10 == 0:
                print(f"  {ok}/{len(frames)}")

        except Exception as e:
            print(f"  {frame_id[:8]} failed: {e}")

    print(f"\n{ok} frames segmented")
    conn.close()


if __name__ == '__main__':
    main()