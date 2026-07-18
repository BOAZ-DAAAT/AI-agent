---
document_id: sales-orders
doc_type: analysis_query_rule
query_type: sales_orders
title: Olist Sales and Orders Query Rules
language: en
version: "1.0"
business_entities: [orders, customers, payments, order_items]
source_tables: [customers, orders, order_items, order_payments]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Sales Orders Query Rules

## Definition

Sales orders analysis measures order volume, item volume, gross merchandise value, freight, and order lifecycle status using Olist order header and item/payment detail tables. The default order grain is one row per `orders.order_id`.

## Supported Intents

- Count orders by time period, status, customer region, seller, product, or category.
- Calculate gross sales from `order_items.price`.
- Calculate freight from `order_items.freight_value`.
- Compare item count, order count, and average order value.
- Analyze order status distribution from `orders.order_status`.
- Track created, approved, shipped, delivered, and estimated delivery milestones.

## Default Metrics

- `COUNT(DISTINCT orders.order_id)` for order count.
- `COUNT(*)` from `order_items` only for item-line count.
- `SUM(order_items.price)` for merchandise sales before freight.
- `SUM(order_items.price + order_items.freight_value)` for merchandise plus freight.
- `SUM(order_payments.payment_value)` only for payment-collected value when explicitly requested.
- `AVG(order_items.price)` for average item price.
- `SUM(order_items.price) / COUNT(DISTINCT orders.order_id)` for average merchandise value per order.

## Entity Grain

- Default grain: `orders.order_id`.
- Item grain: `order_items.order_id`, `order_items.order_item_id`.
- Customer grain, when requested: `customers.customer_unique_id`.
- Seller grain, when requested: `order_items.seller_id`.
- Product grain, when requested: `order_items.product_id`.

## Time Basis

- Use `orders.order_purchase_timestamp` as the default sales/order time.
- Use `orders.order_approved_at` only for approval timing questions.
- Use `orders.order_delivered_customer_date` only for delivered-date questions.
- Use `order_items.shipping_limit_date` only for seller shipping deadline questions.
- Do not use `order_reviews.review_creation_date` for sales timing.

## Required Tables

- `orders`
- `order_items` when sales amount, item count, freight, product, seller, or category is needed.
- `order_payments` only when payment value or payment method is needed.
- `customers` only when customer identity or customer region is needed.
- `products` only when product attributes or product category are needed.
- `product_category_name_translation` only when English category names are needed.
- `sellers` only when seller location is needed.

## Join Constraints

- Join `orders` to `order_items` on `orders.order_id = order_items.order_id`.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id`.
- Join `order_items` to `products` on `order_items.product_id = products.product_id`.
- Join `products` to `product_category_name_translation` on `products.product_category_name = product_category_name_translation.product_category_name`.
- Join `order_items` to `sellers` on `order_items.seller_id = sellers.seller_id`.
- Join `orders` to `order_payments` on `orders.order_id = order_payments.order_id`.
- Avoid joining `order_items` and `order_payments` directly without first aggregating at `order_id`; both tables can have multiple rows per order and can multiply monetary totals.
- Count orders with `COUNT(DISTINCT orders.order_id)` after any join to item, payment, product, review, or seller tables.
- Aggregate `order_items.price` at item grain before joining to payment rows if both item and payment metrics are requested.
- Do not use `customers.customer_id` as a repeat-customer identifier; it is an order-level join key.
- Do not join `geolocation` unless the query explicitly asks for coordinates or zip-prefix geography.

## Status And Null Rules

- Use `orders.order_status` as the only order status field.
- Treat `order_status = 'delivered'` as delivered order status; do not infer delivery from non-null dates alone unless status handling is requested.
- Filter to delivered orders for completed-sales reporting when the user asks for completed, delivered, fulfilled, or final sales.
- Do not exclude non-delivered statuses for general order intake or order creation questions.
- Exclude rows with null `orders.order_purchase_timestamp` when grouping by purchase date.
- Exclude rows with null `order_items.price` from price sums and averages.
- Exclude rows with null `order_items.freight_value` from freight sums and averages.
- Keep null `product_category_name` as unknown category instead of dropping it when the query asks for all categories.
- Keep zero or one-installment payments as valid payment records; do not drop them as null-like values.
- Do not coalesce missing delivery dates to the estimated delivery date.

## Clarify When

- The user says "sales" but does not indicate whether freight should be included.
- The user says "revenue" but does not indicate whether to use item price or payment value.
- The user asks for "valid orders" without defining which statuses are valid.
- The user asks for "recent sales" without a date range.
- Ambiguous query example: "Show total sales by state."
- Ambiguous query example: "Which orders count as successful?"
- Ambiguous query example: "Compare recent order performance."

## Prohibited Interpretations

- Do not treat `order_items` row count as order count.
- Do not treat `order_payments.payment_value` as item price.
- Do not add `order_items.price` to `order_payments.payment_value`; they are alternate monetary views.
- Do not infer net revenue, margin, profit, discounts, refunds, taxes, or fees; those fields are not in the schema.
- Do not treat `customer_id` as a stable customer identifier across multiple orders.
- Do not assume status values outside the values present in `orders.order_status`.
- Do not use review dates as purchase dates.
- Do not infer SKU quantity from `order_item_id`; it is an item sequence within an order.
- Do not use geolocation latitude/longitude for sales region unless coordinates are explicitly requested.
- Do not assume currency conversion or inflation adjustment.

## Unsupported Requests

- Profit, margin, cost of goods sold, seller commission, discounts, refunds, or tax analysis.
- Inventory, stockout, or fulfillment capacity analysis.
- Customer acquisition source or marketing campaign sales attribution.
- Real-time order status beyond the stored Olist tables.
- SKU-level unit quantity when duplicate item rows are not enough to answer the request.

## Limitations

- The schema has no cost, margin, discount, tax, refund, or currency conversion fields.
- `order_payments` can contain multiple payment rows for one order.
- `order_items` can contain multiple item rows for one order.
- `customers.customer_id` maps to an order-level customer record, not a stable person.
- Status values and date completeness depend on the loaded Olist data.

## Positive Examples

- "Count distinct orders by month using purchase timestamp."
- "Show delivered merchandise sales by seller state."
- "Compare order count and item price sum by product category."
- "Calculate average merchandise value per order for 2018."
- "Break down order status counts by customer state."

## Negative Examples

- "Calculate profit by order."
- "Show discounts by product category."
- "Use review date as the order date."
- "Count item rows as total orders."
- "Estimate refunds from payment differences."
