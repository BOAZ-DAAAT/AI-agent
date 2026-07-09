CREATE DATABASE IF NOT EXISTS olist_ecommerce CHARACTER SET utf8mb4;
USE olist_ecommerce;

CREATE TABLE customers (
  customer_id VARCHAR(32) PRIMARY KEY,
  customer_unique_id VARCHAR(32),
  customer_zip_code_prefix INT,
  customer_city VARCHAR(64),
  customer_state CHAR(2)
) CHARACTER SET utf8mb4;

CREATE TABLE geolocation (
  geolocation_zip_code_prefix INT,
  geolocation_lat DOUBLE,
  geolocation_lng DOUBLE,
  geolocation_city VARCHAR(64),
  geolocation_state CHAR(2)
) CHARACTER SET utf8mb4;

CREATE TABLE order_items (
  order_id VARCHAR(32),
  order_item_id INT,
  product_id VARCHAR(32),
  seller_id VARCHAR(32),
  shipping_limit_date DATETIME,
  price DECIMAL(10,2),
  freight_value DECIMAL(10,2),
  PRIMARY KEY (order_id, order_item_id)
) CHARACTER SET utf8mb4;

CREATE TABLE order_payments (
  order_id VARCHAR(32),
  payment_sequential INT,
  payment_type VARCHAR(20),
  payment_installments INT,
  payment_value DECIMAL(10,2)
) CHARACTER SET utf8mb4;

CREATE TABLE order_reviews (
  review_id VARCHAR(32),
  order_id VARCHAR(32),
  review_score INT,
  review_comment_title VARCHAR(255),
  review_comment_message TEXT,
  review_creation_date DATETIME,
  review_answer_timestamp DATETIME
) CHARACTER SET utf8mb4;

CREATE TABLE orders (
  order_id VARCHAR(32) PRIMARY KEY,
  customer_id VARCHAR(32),
  order_status VARCHAR(20),
  order_purchase_timestamp DATETIME,
  order_approved_at DATETIME NULL,
  order_delivered_carrier_date DATETIME NULL,
  order_delivered_customer_date DATETIME NULL,
  order_estimated_delivery_date DATETIME NULL
) CHARACTER SET utf8mb4;

CREATE TABLE products (
  product_id VARCHAR(32) PRIMARY KEY,
  product_category_name VARCHAR(64) NULL,
  product_name_lenght INT NULL,
  product_description_lenght INT NULL,
  product_photos_qty INT NULL,
  product_weight_g INT NULL,
  product_length_cm INT NULL,
  product_height_cm INT NULL,
  product_width_cm INT NULL
) CHARACTER SET utf8mb4;

CREATE TABLE sellers (
  seller_id VARCHAR(32) PRIMARY KEY,
  seller_zip_code_prefix INT,
  seller_city VARCHAR(64),
  seller_state CHAR(2)
) CHARACTER SET utf8mb4;

CREATE TABLE product_category_name_translation (
  product_category_name VARCHAR(64) PRIMARY KEY,
  product_category_name_english VARCHAR(64)
) CHARACTER SET utf8mb4;