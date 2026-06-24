import os
import pathlib
import psycopg2
import mlflow
from dotenv import load_dotenv

p = pathlib.Path('/opt/airflow/.env')
load_dotenv(str(p) if p.exists() else None)


def detection_stats(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT class_name, COUNT(*) as n,
               AVG(confidence), MIN(confidence), MAX(confidence)
        FROM detections
        GROUP BY class_name
        ORDER BY n DESC
    """)
    return cur.fetchall()


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow_local.db")
    mlflow.set_experiment("perception-av-baseline")

    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    stats = detection_stats(conn)
    conn.close()

    with mlflow.start_run(run_name="yolo26n-pretrained-baseline"):
        mlflow.log_param("model", "yolo26n")
        mlflow.log_param("weights", "pretrained-coco")
        mlflow.log_param("dataset", "nuscenes-mini")
        mlflow.log_param("frames", 404)

        total = 0
        for class_name, count, avg_conf, min_conf, max_conf in stats:
            c = class_name.replace(' ', '_')
            mlflow.log_metric(f"{c}_count", count)
            mlflow.log_metric(f"{c}_avg_conf", round(avg_conf, 4))
            mlflow.log_metric(f"{c}_min_conf", round(min_conf, 4))
            total += count

        mlflow.log_metric("total_detections", total)
        mlflow.log_metric("detections_per_frame", round(total / 404, 2))

        mlflow.set_tag("status", "production")
        mlflow.set_tag("version", "1.0")

        print(f"run: {mlflow.active_run().info.run_id}")
        print(f"detections: {total}")


if __name__ == '__main__':
    main()