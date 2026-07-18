---
document_id: delivery-delay
doc_type: analysis_query_rule
query_type: delivery_delay
title: Olist Delivery Delay Query Rules
language: en
version: "1.0"
business_entities: [orders, order_items, customers, sellers]
source_tables: [orders, order_items, customers, sellers]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Delivery Delay Query Rules

## Definition

Delivery delay analysis compares actual customer delivery time with estimated delivery date, shipping deadline, purchase time, approval time, or carrier handoff time using date columns in `orders` and `order_items`.

## Supported Intents

- Identify late deliveries against estimated delivery date.
- Measure delivery duration from purchase to customer delivery.
- Measure approval-to-delivery or carrier-to-delivery duration.
- Compare delay rates by customer region, seller region, category, or seller.
- Analyze seller shipping deadline misses using `order_items.shipping_limit_date`.
- Rank orders, sellers, categories, or regions by average delay days.

## Default Metrics

- Late delivery flag: `orders.order_delivered_customer_date > orders.order_estimated_delivery_date`.
- Delay days: difference between `order_delivered_customer_date` and `order_estimated_delivery_date`.
- Delivery duration days: difference between `order_delivered_customer_date` and `order_purchase_timestamp`.
- Carrier transit days: difference between `order_delivered_customer_date` and `order_delivered_carrier_date`.
- Shipping deadline miss flag: `order_items.shipping_limit_date < orders.order_delivered_carrier_date` when seller shipping delay is requested.
- Late rate: late delivered orders divided by delivered orders with both actual and estimated dates.

## Entity Grain

- Default grain: `orders.order_id`.
- Seller shipping deadline grain: `order_items.order_id`, `order_items.order_item_id`, or `order_items.seller_id` depending on the question.
- Customer region grain: `customers.customer_state` or `customers.customer_city`.
- Seller grain: `order_items.seller_id`.
- Category grain: translated category after item-product-category joins.

## Time Basis

- Use `orders.order_purchase_timestamp` for order cohort timing.
- Use `orders.order_estimated_delivery_date` as the default promised delivery basis.
- Use `orders.order_delivered_customer_date` as the actual delivery basis.
- Use `orders.order_delivered_carrier_date` for carrier handoff and transit questions.
- Use `order_items.shipping_limit_date` only for seller shipping deadline analysis.

## Required Tables

- `orders`
- `order_items` only for seller, product, category, freight, or shipping-limit analysis.
- `customers` only for customer region analysis.
- `sellers` only for seller region analysis.
- `products` only for product/category delay analysis.
- `product_category_name_translation` only when English category names are requested.

## Join Constraints

- Join `orders` to `order_items` on `orders.order_id = order_items.order_id` only when item, seller, category, or shipping-limit fields are needed.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id`.
- Join `order_items` to `sellers` on `order_items.seller_id = sellers.seller_id`.
- Join `order_items` to `products` on `order_items.product_id = products.product_id`.
- Join category translation through `products.product_category_name`.
- Count delayed orders with `COUNT(DISTINCT orders.order_id)` after joining item-level tables.
- For seller shipping-limit misses, calculate at item grain before rolling up to seller.
- Do not join payments for delivery delay unless payment timing/value is explicitly requested.
- Do not join reviews unless satisfaction impact is explicitly requested.
- Do not join geolocation unless coordinate or zip-prefix distance analysis is requested.

## Status And Null Rules

- Restrict default delivery delay metrics to orders with `orders.order_status = 'delivered'`.
- Require non-null `order_delivered_customer_date` and `order_estimated_delivery_date` for late-delivery flags.
- Exclude null actual delivery dates from average delay duration unless the query asks about undelivered orders.
- Do not treat undelivered orders as delayed by default.
- Do not coalesce missing actual delivery dates to current date.
- Do not coalesce missing estimated delivery dates to purchase date.
- Require non-null `order_delivered_carrier_date` for carrier transit calculations.
- Require non-null `order_items.shipping_limit_date` for seller shipping deadline calculations.
- Negative delay days mean early delivery and must not be converted to zero unless the user requests lateness-only magnitude.
- Keep canceled, unavailable, shipped, or processing statuses separate when the query asks about non-delivery status.

## Clarify When

- The user says "delay" but does not say estimated delivery delay, seller shipping delay, or carrier transit delay.
- The user asks for "late orders" but includes undelivered orders without a rule for current-date comparison.
- The user asks for "delivery performance" without choosing speed, late rate, or status completion.
- The user asks for "problem sellers" without specifying delay threshold or metric.
- Ambiguous query example: "Which deliveries were delayed?"
- Ambiguous query example: "Find bad sellers by shipping."
- Ambiguous query example: "Analyze delivery performance by region."

## Prohibited Interpretations

- Do not infer delay from review text alone.
- Do not use `shipping_limit_date` as the customer promised delivery date.
- Do not use `order_delivered_carrier_date` as customer delivery date.
- Do not include undelivered orders in actual delivery duration by default.
- Do not count item rows as delayed orders.
- Do not use approval time as purchase time unless approval delay is requested.
- Do not infer logistics carrier identity; no carrier name field exists.
- Do not infer distance without explicit geolocation use and zip-prefix caveats.
- Do not use review creation date as delivery date.
- Do not treat early delivery as missing or delayed.

## Unsupported Requests

- Carrier-specific delay analysis; no carrier identifier exists.
- Exact route distance or travel time without external mapping data.
- Warehouse processing delay beyond available timestamps.
- Real-time delivery tracking.
- Delay cause classification unless based only on available status/date fields.

## Limitations

- Delivery delay can be measured only where actual and estimated delivery dates exist.
- Seller shipping-limit analysis is item-level and may differ from order-level late delivery.
- Zip-prefix geolocation is approximate and may have duplicate prefixes.
- The schema does not identify logistics carriers or delay reasons.
- Status and timestamp completeness control denominator size.

## Positive Examples

- "Calculate late delivery rate for delivered orders."
- "Show average delay days by customer_state."
- "Rank sellers by item shipping deadline miss rate."
- "Compare purchase-to-delivery days by category."
- "Find delivered orders where actual delivery exceeded estimated delivery."

## Negative Examples

- "Use review comments to decide which orders were late."
- "Count shipped but undelivered orders as late using today's date."
- "Use shipping_limit_date as promised customer delivery date."
- "Rank carriers by delay."
- "Count item rows as delayed orders."
