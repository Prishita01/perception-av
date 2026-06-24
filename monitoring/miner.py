import os
import pathlib
import psycopg2
from dotenv import load_dotenv

p = pathlib.Path('/opt/airflow/.env')
load_dotenv(str(p) if p.exists() else None)


def mine_low_confidence(conn):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retraining_queue (frame_id, mining_strategy, priority_score, reason)
        SELECT
            frame_id,
            'confidence_miner',
            1.0 - AVG(confidence),
            CONCAT('avg_conf=', ROUND(AVG(confidence)::numeric, 3))
        FROM detections
        GROUP BY frame_id
        HAVING AVG(confidence) < 0.4
        ON CONFLICT (frame_id, mining_strategy) DO NOTHING
    """)
    n = cur.rowcount
    conn.commit()
    cur.close()
    print(f"confidence_miner: {n} frames")
    return n


def mine_traffic_light_failures(conn):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retraining_queue (frame_id, mining_strategy, priority_score, reason)
        SELECT
            frame_id,
            'class_miner',
            1.0 - AVG(confidence),
            CONCAT('tl_conf=', ROUND(AVG(confidence)::numeric, 3))
        FROM detections
        WHERE class_name = 'traffic light' AND confidence < 0.35
        GROUP BY frame_id
        ON CONFLICT (frame_id, mining_strategy) DO NOTHING
    """)
    n = cur.rowcount
    conn.commit()
    cur.close()
    print(f"class_miner (traffic lights): {n} frames")
    return n


def mine_dense_scenes(conn):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO retraining_queue (frame_id, mining_strategy, priority_score, reason)
        SELECT
            frame_id,
            'density_miner',
            COUNT(*) * (1.0 - AVG(confidence)),
            CONCAT(COUNT(*), ' dets, avg_conf=', ROUND(AVG(confidence)::numeric, 3))
        FROM detections
        GROUP BY frame_id
        HAVING COUNT(*) > 8 AND AVG(confidence) < 0.5
        ON CONFLICT (frame_id, mining_strategy) DO NOTHING
    """)
    n = cur.rowcount
    conn.commit()
    cur.close()
    print(f"density_miner: {n} frames")
    return n


def queue_summary(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT mining_strategy, COUNT(*) as frames,
               ROUND(AVG(priority_score)::numeric, 3) as avg_priority
        FROM retraining_queue
        GROUP BY mining_strategy
        ORDER BY avg_priority DESC
    """)
    rows = cur.fetchall()
    cur.close()
    return rows


def main():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )

    total = sum([
        mine_low_confidence(conn),
        mine_traffic_light_failures(conn),
        mine_dense_scenes(conn),
    ])

    print(f"\n{total} frames added to retraining queue")
    print("\nqueue summary:")
    for strategy, count, avg_p in queue_summary(conn):
        print(f"  {strategy:<20} {count} frames  avg_priority={avg_p}")

    cur = conn.cursor()
    cur.execute("""
        SELECT frame_id, mining_strategy, priority_score, reason
        FROM retraining_queue
        ORDER BY priority_score DESC
        LIMIT 5
    """)
    print("\ntop 5 by priority:")
    for row in cur.fetchall():
        print(f"  {row[0][:16]}  [{row[1]}]  score={row[2]:.3f}  {row[3]}")
    cur.close()
    conn.close()


if __name__ == '__main__':
    main()