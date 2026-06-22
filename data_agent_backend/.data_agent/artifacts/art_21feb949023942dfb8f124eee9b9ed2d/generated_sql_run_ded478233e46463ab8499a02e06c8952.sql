SELECT
  COALESCE(pcnt.product_category_name_english, p.product_category_name, 'Unknown') AS category_name,
  COUNT(DISTINCT o.order_id) AS total_orders,
  AVG(CASE WHEN orv.review_score IS NOT NULL THEN orv.review_score ELSE NULL END) AS avg_review_score,
  AVG(CASE WHEN o.order_delivered_customer_date IS NOT NULL AND o.order_purchase_timestamp IS NOT NULL THEN DATEDIFF(o.order_delivered_customer_date, o.order_purchase_timestamp) ELSE NULL END) AS avg_delivery_days
FROM orders AS o
JOIN order_items AS oi
  ON o.order_id = oi.order_id
JOIN products AS p
  ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation AS pcnt
  ON p.product_category_name = pcnt.product_category_name
LEFT JOIN order_reviews AS orv
  ON o.order_id = orv.order_id
GROUP BY
  category_name
ORDER BY
  total_orders DESC;