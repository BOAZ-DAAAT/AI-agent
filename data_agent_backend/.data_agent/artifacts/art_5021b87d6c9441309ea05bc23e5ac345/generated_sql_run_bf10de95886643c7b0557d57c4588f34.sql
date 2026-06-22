SELECT
  'order_id' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_id) AS non_null_rows,
  COUNT(DISTINCT order_id) AS distinct_values,
  (COUNT(*) - COUNT(order_id)) / COUNT(*) * 100 AS null_percentage,
  NULL AS min_value,
  NULL AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'customer_id' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(customer_id) AS non_null_rows,
  COUNT(DISTINCT customer_id) AS distinct_values,
  (COUNT(*) - COUNT(customer_id)) / COUNT(*) * 100 AS null_percentage,
  NULL AS min_value,
  NULL AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_status' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_status) AS non_null_rows,
  COUNT(DISTINCT order_status) AS distinct_values,
  (COUNT(*) - COUNT(order_status)) / COUNT(*) * 100 AS null_percentage,
  NULL AS min_value,
  NULL AS max_value,
  NULL AS avg_value,
  (SELECT GROUP_CONCAT(CONCAT(order_status, ' (', count_status, ')') ORDER BY count_status DESC SEPARATOR '; ') FROM (SELECT order_status, COUNT(*) AS count_status FROM orders GROUP BY order_status ORDER BY count_status DESC LIMIT 5) AS top_status) AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_purchase_timestamp' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_purchase_timestamp) AS non_null_rows,
  COUNT(DISTINCT order_purchase_timestamp) AS distinct_values,
  (COUNT(*) - COUNT(order_purchase_timestamp)) / COUNT(*) * 100 AS null_percentage,
  MIN(order_purchase_timestamp) AS min_value,
  MAX(order_purchase_timestamp) AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_approved_at' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_approved_at) AS non_null_rows,
  COUNT(DISTINCT order_approved_at) AS distinct_values,
  (COUNT(*) - COUNT(order_approved_at)) / COUNT(*) * 100 AS null_percentage,
  MIN(order_approved_at) AS min_value,
  MAX(order_approved_at) AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_delivered_carrier_date' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_delivered_carrier_date) AS non_null_rows,
  COUNT(DISTINCT order_delivered_carrier_date) AS distinct_values,
  (COUNT(*) - COUNT(order_delivered_carrier_date)) / COUNT(*) * 100 AS null_percentage,
  MIN(order_delivered_carrier_date) AS min_value,
  MAX(order_delivered_carrier_date) AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_delivered_customer_date' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_delivered_customer_date) AS non_null_rows,
  COUNT(DISTINCT order_delivered_customer_date) AS distinct_values,
  (COUNT(*) - COUNT(order_delivered_customer_date)) / COUNT(*) * 100 AS null_percentage,
  MIN(order_delivered_customer_date) AS min_value,
  MAX(order_delivered_customer_date) AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders
UNION ALL
SELECT
  'order_estimated_delivery_date' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(order_estimated_delivery_date) AS non_null_rows,
  COUNT(DISTINCT order_estimated_delivery_date) AS distinct_values,
  (COUNT(*) - COUNT(order_estimated_delivery_date)) / COUNT(*) * 100 AS null_percentage,
  MIN(order_estimated_delivery_date) AS min_value,
  MAX(order_estimated_delivery_date) AS max_value,
  NULL AS avg_value,
  NULL AS top_5_values
FROM orders;