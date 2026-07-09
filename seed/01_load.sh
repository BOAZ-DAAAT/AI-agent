#!/bin/bash
set -e
DIR="/docker-entrypoint-initdb.d/csv"
M="mysql --local-infile=1 -uroot -p${MYSQL_ROOT_PASSWORD} olist_ecommerce"
OPT="FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED BY '\"' LINES TERMINATED BY '\n' IGNORE 1 LINES"

echo "[seed] customers";  $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_customers_dataset.csv' INTO TABLE customers $OPT (customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state);"

echo "[seed] geolocation"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_geolocation_dataset.csv' INTO TABLE geolocation $OPT (geolocation_zip_code_prefix, geolocation_lat, geolocation_lng, geolocation_city, geolocation_state);"

echo "[seed] order_items"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_order_items_dataset.csv' INTO TABLE order_items $OPT (order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value);"

echo "[seed] order_payments"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_order_payments_dataset.csv' INTO TABLE order_payments $OPT (order_id, payment_sequential, payment_type, payment_installments, payment_value);"

echo "[seed] order_reviews"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_order_reviews_dataset.csv' INTO TABLE order_reviews $OPT (review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp);"

echo "[seed] orders"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_orders_dataset.csv' INTO TABLE orders $OPT (order_id, customer_id, order_status, order_purchase_timestamp, @approved, @carrier, @delivered, order_estimated_delivery_date) SET order_approved_at=NULLIF(@approved,''), order_delivered_carrier_date=NULLIF(@carrier,''), order_delivered_customer_date=NULLIF(@delivered,'');"

echo "[seed] products"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_products_dataset.csv' INTO TABLE products $OPT (product_id, @cat, @nl, @dl, @ph, @wt, @ln, @ht, @wd) SET product_category_name=NULLIF(@cat,''), product_name_lenght=NULLIF(@nl,''), product_description_lenght=NULLIF(@dl,''), product_photos_qty=NULLIF(@ph,''), product_weight_g=NULLIF(@wt,''), product_length_cm=NULLIF(@ln,''), product_height_cm=NULLIF(@ht,''), product_width_cm=NULLIF(@wd,'');"

echo "[seed] sellers"; $M -e "LOAD DATA LOCAL INFILE '$DIR/olist_sellers_dataset.csv' INTO TABLE sellers $OPT (seller_id, seller_zip_code_prefix, seller_city, seller_state);"

echo "[seed] category_translation"; $M -e "LOAD DATA LOCAL INFILE '$DIR/product_category_name_translation.csv' INTO TABLE product_category_name_translation $OPT (product_category_name, product_category_name_english);"

echo "[seed] done."