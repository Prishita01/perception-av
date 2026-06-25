select
    d.id,
    d.frame_id,
    d.class_name,
    d.confidence,
    d.bbox_x_min,
    d.bbox_y_min,
    d.bbox_x_max,
    d.bbox_y_max,
    f.timestamp,
    f.scene_token,
    (d.bbox_x_max - d.bbox_x_min) * (d.bbox_y_max - d.bbox_y_min) as bbox_area
from detections d
join frames f on d.frame_id = f.frame_id