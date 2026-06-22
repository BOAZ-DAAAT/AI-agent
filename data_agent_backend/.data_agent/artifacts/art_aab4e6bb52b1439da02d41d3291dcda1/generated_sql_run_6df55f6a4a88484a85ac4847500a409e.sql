WITH CategoryOrderData AS (
    SELECT
        pct.product_category_name_english AS category_name,
        oi.order_id,
        o.order_purchase_timestamp,
        o.order_delivered_customer_date
    FROM order_items oi
    JOIN products p ON oi.product_id = p.product_id
    LEFT JOIN product_category_name_translation pct ON p.product_category_name = pct.product_category_name
    JOIN orders o ON oi.order_id = o.order_id
    WHERE o.order_delivered_customer_date IS NOT NULL
)
SELECT
    COALESCE(cod.category_name, 'Unknown') AS category,
    COUNT(DISTINCT cod.order_id) AS total_orders,
    AVG(orr.review_score) AS average_review_score,
    AVG(DATEDIFF(cod.order_delivered_customer_date, cod.order_purchase_timestamp)) AS average_delivery_days
FROM CategoryOrderData cod
LEFT JOIN order_reviews orr ON cod.order_id = orr.order_id
GROUP BY category
ORDER BY total_orders DESC;