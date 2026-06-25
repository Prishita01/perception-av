select
    frame_id,
    count(*) as detection_count,
    round(avg(confidence)::numeric, 3) as avg_confidence,
    sum(case when class_name = 'traffic light' then 1 else 0 end) as traffic_light_count,
    sum(case when class_name = 'person' then 1 else 0 end) as person_count,
    sum(case when class_name = 'car' then 1 else 0 end) as car_count,
    timestamp
from {{ ref('stg_detections') }}
group by frame_id, timestamp
order by timestamp asc