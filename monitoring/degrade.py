import os
import pathlib
import random
import psycopg2
from dotenv import load_dotenv

p = pathlib.Path('/opt/airflow/.env')
load_dotenv(str(p) if p.exists() else None)


def db_conn():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )

def main():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute(" SELECT frame_id FROM frames ORDER BY timestamp DESC LIMIT 100")
    frames = [row[0] for row in cur.fetchall()]
    print(f"degrading {len(frames)} frames")

    for frame_id in frames:
        cur.execute(
            "UPDATE detections SET confidence = confidence * %s WHERE frame_id = %s",
            (random.uniform(0.3, 0.5), frame_id)
        )

    conn.commit()
    cur.close()
    conn.close()

    conn2 = db_conn()
    cur2 = conn2.cursor()
    cur2.execute("""
        SELECT class_name, ROUND(AVG(confidence)::numeric, 3)
        FROM detections
        GROUP BY class_name
        ORDER BY AVG(confidence) ASC
    """)
    print("\nconfidence after degradation:")
    for row in cur2.fetchall():
        print(f"  {row[0]:<16} {row[1]}")
    conn2.close()


if __name__ == '__main__':
    main()