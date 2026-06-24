CREATE TABLE IF NOT EXISTS frames (
    frame_id       VARCHAR(255) PRIMARY KEY,
    scene_token    VARCHAR(255) NOT NULL,
    timestamp      BIGINT NOT NULL,
    camera_channel VARCHAR(50) NOT NULL,
    s3_key         VARCHAR(500) NOT NULL,
    depth_s3_key   VARCHAR(500),
    depth_processed BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detections (
    id          SERIAL PRIMARY KEY,
    frame_id    VARCHAR(255) NOT NULL REFERENCES frames(frame_id),
    class_id    INTEGER NOT NULL,
    class_name  VARCHAR(50) NOT NULL,
    confidence  FLOAT NOT NULL,
    bbox_x_min  FLOAT,
    bbox_y_min  FLOAT,
    bbox_x_max  FLOAT,
    bbox_y_max  FLOAT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_detections_frame ON detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_class ON detections(class_name);
CREATE INDEX IF NOT EXISTS idx_detections_conf  ON detections(confidence);

CREATE TABLE IF NOT EXISTS masks (
    id           SERIAL PRIMARY KEY,
    frame_id     VARCHAR(255) NOT NULL REFERENCES frames(frame_id),
    detection_id INTEGER NOT NULL REFERENCES detections(id),
    class_name   VARCHAR(50) NOT NULL,
    rle_mask     TEXT NOT NULL,
    area         INTEGER NOT NULL,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(detection_id)
);

CREATE INDEX IF NOT EXISTS idx_masks_frame ON masks(frame_id);
CREATE INDEX IF NOT EXISTS idx_masks_class ON masks(class_name);

CREATE TABLE IF NOT EXISTS retraining_queue (
    id               SERIAL PRIMARY KEY,
    frame_id         VARCHAR(255) NOT NULL REFERENCES frames(frame_id),
    mining_strategy  VARCHAR(50) NOT NULL,
    priority_score   FLOAT NOT NULL,
    reason           TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(frame_id, mining_strategy)
);

CREATE INDEX IF NOT EXISTS idx_queue_frame    ON retraining_queue(frame_id);
CREATE INDEX IF NOT EXISTS idx_queue_priority ON retraining_queue(priority_score DESC);