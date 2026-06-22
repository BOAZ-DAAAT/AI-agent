SELECT
  COALESCE(pct.product_category_name_english, p.product_category_name) AS category_name,
  COUNT(DISTINCT o.order_id) AS total_orders,
  AVG(orv.review_score) AS average_review_score,
  AVG(DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp)) AS average_delivery_days
FROM orders AS o
JOIN order_items AS oi
  ON o.order_id = oi.order_id
JOIN products AS p
  ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation AS pct
  ON p.product_category_name = pct.product_category_name
LEFT JOIN order_reviews AS orv
  ON o.order_id = orv.order_id
WHERE
  o.order_delivered_customer_date IS NOT NULL AND o.order_purchase_timestamp IS NOT NULL
GROUP BY
  category_name
ORDER BY
  total_orders DESC;