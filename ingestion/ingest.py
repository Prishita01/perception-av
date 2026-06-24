import os
import boto3
import psycopg2
from dotenv import load_dotenv
from nuscenes.nuscenes import NuScenes

load_dotenv()

def load_nuscenes():
    return NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)

def upload_frame(local_path, s3_key):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, os.getenv('S3_BUCKET_NAME'), s3_key)
    return s3_key

def insert_frame(conn, frame):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO frames (frame_id, scene_token, timestamp, camera_channel, s3_key)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (frame_id) DO NOTHING
    """, (
        frame['frame_id'], frame['scene_token'], frame['timestamp'],
        frame['camera_channel'], frame['s3_key'],
    ))
    conn.commit()
    cur.close()

def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )

    nusc = load_nuscenes()
    ok, err = 0, 0

    for sample in nusc.sample:
        cam_token = sample['data']['CAM_FRONT']
        cam = nusc.get('sample_data', cam_token)
        frame_id = cam['token']
        local_path = os.path.join('data/raw', cam['filename'])
        s3_key = f"frames/{sample['scene_token']}/{frame_id}.jpg"

        try:
            upload_frame(local_path, s3_key)
            insert_frame(conn, {
                'frame_id': frame_id,
                'scene_token': sample['scene_token'],
                'timestamp': cam['timestamp'],
                'camera_channel': 'CAM_FRONT',
                's3_key': s3_key,
            })
            ok += 1
        except Exception as e:
            print(f"  {frame_id[:8]} failed: {e}")
            err += 1

    print(f"{ok} frames ingested, {err} errors")
    conn.close()


if __name__ == '__main__':
    main()