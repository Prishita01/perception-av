select
    class_name,
    count(*)                                    as total_detections,
    round(avg(confidence)::numeric, 3)          as avg_confidence,
    round(min(confidence)::numeric, 3)          as min_confidence,
    round(max(confidence)::numeric, 3)          as max_confidence,
    round(avg(bbox_area)::numeric, 0)           as avg_bbox_area
from {{ ref('stg_detections') }}
group by class_name
order by total_detections desc