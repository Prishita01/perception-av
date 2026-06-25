select
    mining_strategy,
    count(*) as frame_count,
    round(avg(priority_score)::numeric, 3) as avg_priority,
    round(max(priority_score)::numeric, 3) as max_priority
from retraining_queue
group by mining_strategy
order by avg_priority desc