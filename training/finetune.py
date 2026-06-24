import os
import pathlib
import mlflow
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

WEIGHTS_PATH = 'runs/detect/training/runs/hard_case_ft/weights/best.pt'
CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic cone']


def per_class_ap(results, classes):
    out = {}
    for i, name in enumerate(classes):
        try:
            out[name] = round(float(results.box.ap50[i]), 4)
        except Exception:
            out[name] = 0.0
    return out


def main():
    data_yaml = str(pathlib.Path('training/hard_cases.yaml').absolute())

    mlflow.set_tracking_uri("sqlite:///mlflow_local.db")
    mlflow.set_experiment("perception-av-finetuning")

    base_model = YOLO("yolo26n.pt")
    base_results = base_model.val(data=data_yaml, device='mps', verbose=False)
    base_map50 = round(float(base_results.box.map50), 4)
    base_map   = round(float(base_results.box.map), 4)
    base_ap    = per_class_ap(base_results, CLASSES)

    print(f"baseline  mAP50={base_map50}  mAP50-95={base_map}")

    train_model = YOLO("yolo26n.pt")
    train_model.train(
        data=data_yaml,
        epochs=20,
        imgsz=640,
        device='mps',
        batch=8,
        patience=10,
        save=True,
        project='training/runs',
        name='hard_case_ft',
        exist_ok=True,
        verbose=False,
    )

    ft_model   = YOLO(WEIGHTS_PATH)
    ft_results = ft_model.val(data=data_yaml, device='mps', verbose=False)
    ft_map50   = round(float(ft_results.box.map50), 4)
    ft_map     = round(float(ft_results.box.map), 4)
    ft_ap      = per_class_ap(ft_results, CLASSES)

    print(f"finetuned mAP50={ft_map50}  mAP50-95={ft_map}")
    print(f"delta     mAP50={ft_map50-base_map50:+.4f}  mAP50-95={ft_map-base_map:+.4f}")

    with mlflow.start_run(run_name="yolo26n-hard-case-ft-20ep"):
        mlflow.log_param("model", "yolo26n")
        mlflow.log_param("epochs", 20)
        mlflow.log_param("batch", 8)
        mlflow.log_param("frames", 50)
        mlflow.log_param("dataset", "nuscenes-mini-hard-cases")

        mlflow.log_metric("baseline_mAP50", base_map50)
        mlflow.log_metric("baseline_mAP50_95", base_map)
        mlflow.log_metric("ft_mAP50", ft_map50)
        mlflow.log_metric("ft_mAP50_95", ft_map)
        mlflow.log_metric("delta_mAP50", round(ft_map50 - base_map50, 4))
        mlflow.log_metric("delta_mAP50_95", round(ft_map - base_map, 4))

        for name, v in base_ap.items():
            mlflow.log_metric(f"baseline_{name}_AP50", v)
        for name, v in ft_ap.items():
            mlflow.log_metric(f"ft_{name}_AP50", v)

        mlflow.log_artifact(WEIGHTS_PATH, artifact_path='model')
        print(f"run: {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    main()