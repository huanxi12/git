-- DSS warehouse DDL (generic MySQL/PostgreSQL style)
CREATE TABLE IF NOT EXISTS dim_user (
  user_id VARCHAR(64) PRIMARY KEY,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_product (
  product_id VARCHAR(64) PRIMARY KEY,
  product_name TEXT,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dim_category (
  category_id VARCHAR(64) PRIMARY KEY,
  category_name TEXT
);

CREATE TABLE IF NOT EXISTS dim_date (
  date_key DATE PRIMARY KEY,
  year INT,
  month INT,
  day INT,
  weekday INT,
  is_weekend INT
);

CREATE TABLE IF NOT EXISTS bridge_product_category (
  product_id VARCHAR(64),
  category_id VARCHAR(64),
  category_order INT,
  PRIMARY KEY(product_id, category_id, category_order)
);

CREATE TABLE IF NOT EXISTS fact_review_train (
  review_id VARCHAR(64) PRIMARY KEY,
  user_id VARCHAR(64),
  product_id VARCHAR(64),
  review_time TIMESTAMP,
  title TEXT,
  content TEXT,
  clean_text TEXT,
  score INT,
  is_conflict INT,
  etl_batch_id VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  is_deleted INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fact_review_test (
  review_id VARCHAR(64) PRIMARY KEY,
  user_id VARCHAR(64),
  product_id VARCHAR(64),
  review_time TIMESTAMP,
  title TEXT,
  content TEXT,
  clean_text TEXT,
  is_conflict INT,
  etl_batch_id VARCHAR(64),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  is_deleted INT DEFAULT 0
);

CREATE TABLE IF NOT EXISTS fact_prediction (
  review_id VARCHAR(64) PRIMARY KEY,
  pred_score INT,
  pred_sentiment VARCHAR(16),
  risk_level VARCHAR(16),
  risk_tags TEXT,
  model_version VARCHAR(64),
  inference_time TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agg_product_daily (
  product_id VARCHAR(64),
  dt DATE,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  PRIMARY KEY(product_id, dt)
);

CREATE TABLE IF NOT EXISTS agg_category_daily (
  category_id VARCHAR(64),
  dt DATE,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  PRIMARY KEY(category_id, dt)
);

CREATE TABLE IF NOT EXISTS agg_user_profile (
  user_id VARCHAR(64) PRIMARY KEY,
  review_cnt INT,
  avg_score FLOAT,
  low_score_rate FLOAT,
  active_days INT,
  favored_categories TEXT,
  user_tier VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS risk_alert_log (
  alert_id VARCHAR(64) PRIMARY KEY,
  product_id VARCHAR(64),
  dt DATE,
  risk_level VARCHAR(16),
  evidence JSON,
  suggestion TEXT,
  owner VARCHAR(64),
  created_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_train_product_date ON fact_review_train(product_id, review_time);
CREATE INDEX IF NOT EXISTS idx_fact_train_user ON fact_review_train(user_id);
CREATE INDEX IF NOT EXISTS idx_agg_category_date ON agg_category_daily(category_id, dt);
