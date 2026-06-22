SELECT
  pct.product_category_name_english,
  COUNT(oi.order_item_id) AS items_sold
FROM order_items AS oi
JOIN products AS p
  ON oi.product_id = p.product_id
LEFT JOIN product_category_name_translation AS pct
  ON p.product_category_name = pct.product_category_name
GROUP BY
  pct.product_category_name_english
ORDER BY
  items_sold DESC
LIMIT 5;