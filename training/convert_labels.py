import os
import json
import shutil
import pathlib
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.data_classes import Box
from nuscenes.scripts.export_2d_annotations_as_json import post_process_coords
import psycopg2
from dotenv import load_dotenv

load_dotenv()

CATEGORY_MAP = {
    'human.pedestrian.adult':               'person',
    'human.pedestrian.child':               'person',
    'human.pedestrian.construction_worker': 'person',
    'human.pedestrian.personal_mobility':   'person',
    'human.pedestrian.police_officer':      'person',
    'vehicle.car':                          'car',
    'vehicle.truck':                        'truck',
    'vehicle.bus.bendy':                    'bus',
    'vehicle.bus.rigid':                    'bus',
    'vehicle.motorcycle':                   'motorcycle',
    'vehicle.bicycle':                      'bicycle',
    'vehicle.construction':                 'truck',
    'vehicle.trailer':                      'truck',
    'movable_object.trafficcone':           'traffic cone',
    'movable_object.barrier':               None,
    'movable_object.debris':                None,
    'movable_object.pushable_pullable':     None,
    'static_object.bicycle_rack':           None,
}

CLASSES = ['person', 'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'traffic cone']


def project_to_image(nusc, ann_token, cam_data):
    ann    = nusc.get('sample_annotation', ann_token)
    cam_cs = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    cam_ego = nusc.get('ego_pose', cam_data['ego_pose_token'])

    box = Box(ann['translation'], ann['size'], Quaternion(ann['rotation']))
    box.translate(-np.array(cam_ego['translation']))
    box.rotate(Quaternion(cam_ego['rotation']).inverse)
    box.translate(-np.array(cam_cs['translation']))
    box.rotate(Quaternion(cam_cs['rotation']).inverse)

    corners = view_points(box.corners(), np.array(cam_cs['camera_intrinsic']), normalize=True)
    coords  = post_process_coords(list(zip(corners[0].tolist(), corners[1].tolist())))
    if coords is None:
        return None

    x1, y1, x2, y2 = coords
    w, h = cam_data['width'], cam_data['height']
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None

    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def to_yolo_format(x1, y1, x2, y2, img_w, img_h):
    return (
        ((x1 + x2) / 2) / img_w,
        ((y1 + y2) / 2) / img_h,
        (x2 - x1) / img_w,
        (y2 - y1) / img_h,
    )


def priority_frame_ids():
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'), port=os.getenv('DB_PORT'),
        dbname=os.getenv('DB_NAME'), user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
    )
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT frame_id FROM retraining_queue
        ORDER BY frame_id LIMIT 50
    """)
    ids = {row[0] for row in cur.fetchall()}
    conn.close()
    return ids

def main():
    nusc = NuScenes(version='v1.0-mini', dataroot='data/raw', verbose=False)

    out = pathlib.Path('training/hard_cases')
    (out / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    all_samples = list(nusc.sample)
    # deterministic 80/20 split — same split every run
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    val_samples   = all_samples[split_idx:]

    print(f"train: {len(train_samples)} samples  val: {len(val_samples)} samples")

    converted, skipped = 0, 0

    for split_name, samples in [('train', train_samples), ('val', val_samples)]:
        for sample in samples:
            cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            frame_id = cam_data['token']
            img_w, img_h = cam_data['width'], cam_data['height']
            labels = []

            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                cls = CATEGORY_MAP.get(ann['category_name'])
                if cls is None:
                    continue

                coords = project_to_image(nusc, ann_token, cam_data)
                if coords is None:
                    continue

                xc, yc, w, h = to_yolo_format(*coords, img_w, img_h)
                if w > 0.01 and h > 0.01:
                    labels.append(f"{CLASSES.index(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not labels:
                skipped += 1
                continue

            shutil.copy(
                os.path.join('data/raw', cam_data['filename']),
                out / 'images' / split_name / f"{frame_id}.jpg"
            )
            (out / 'labels' / split_name / f"{frame_id}.txt").write_text('\n'.join(labels))
            converted += 1

    print(f"{converted} converted, {skipped} skipped")

    abs_path = str(out.absolute())
    pathlib.Path('training/hard_cases.yaml').write_text(
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    print("dataset config updated")

if __name__ == '__main__':
    main()