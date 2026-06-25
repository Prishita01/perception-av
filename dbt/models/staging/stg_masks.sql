select
    m.id,
    m.frame_id,
    m.class_name,
    m.area,
    f.timestamp,
    f.scene_token
from masks m
join frames f on m.frame_id = f.frame_id