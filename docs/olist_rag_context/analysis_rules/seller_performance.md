---
document_id: seller-performance
doc_type: analysis_query_rule
query_type: seller_performance
title: Olist Seller Performance Query Rules
language: en
version: "1.0"
business_entities: [sellers, orders, order_items, reviews]
source_tables: [sellers, order_items, orders, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Seller Performance Query Rules

## Definition

Seller performance analysis measures order/item volume, merchandise value, freight, delivery timeliness, review satisfaction, and geographic distribution for sellers identified by `order_items.seller_id` and described by `sellers`.

## Supported Intents

- Rank sellers by item count, distinct order count, or merchandise value.
- Compare seller delivery deadline miss rates.
- Compare seller review scores.
- Analyze seller performance by seller state or city.
- Analyze seller category mix.
- Identify sellers with high late delivery, low reviews, or high sales.
- Compare freight or average item price by seller.

## Default Metrics

- `COUNT(DISTINCT orders.order_id)` for seller order count.
- `COUNT(*)` from `order_items` for seller item-line count.
- `SUM(order_items.price)` for seller merchandise value.
- `SUM(order_items.freight_value)` for seller freight.
- `AVG(order_reviews.review_score)` for seller satisfaction when requested.
- Shipping deadline miss rate using `order_items.shipping_limit_date` and `orders.order_delivered_carrier_date`.
- Late delivery rate using `orders.order_delivered_customer_date` and `orders.order_estimated_delivery_date` when customer delivery delay is requested.

## Entity Grain

- Default grain: `order_items.seller_id`.
- Seller location grain: `sellers.seller_state` or `sellers.seller_city`.
- Item grain: `order_items.order_id`, `order_items.order_item_id`.
- Order grain: `orders.order_id`.
- Category grain, when requested: product category joined through `products`.

## Time Basis

- Use `orders.order_purchase_timestamp` for seller sales/order trends.
- Use `order_items.shipping_limit_date` for seller shipping-deadline obligations.
- Use `orders.order_delivered_carrier_date` for seller handoff timing.
- Use `orders.order_delivered_customer_date` for customer delivery completion timing.
- Use review dates only when reviewing response timing or review volume over time.

## Required Tables

- `order_items`
- `sellers`
- `orders` when order status, time, delivery, or distinct order count is needed.
- `order_reviews` only when review satisfaction is requested.
- `products` only when seller category mix is requested.
- `product_category_name_translation` only when English category names are requested.
- `customers` only when customer region for seller orders is requested.

## Join Constraints

- Join `order_items` to `sellers` on `order_items.seller_id = sellers.seller_id`.
- Join `order_items` to `orders` on `order_items.order_id = orders.order_id`.
- Join `orders` to `order_reviews` on `orders.order_id = order_reviews.order_id` only for review metrics.
- Join `order_items` to `products` on `order_items.product_id = products.product_id` for category metrics.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id` for customer geography.
- Count seller orders with `COUNT(DISTINCT orders.order_id)` because one seller can have multiple item rows in an order.
- Use item-line grain for seller merchandise value because seller ownership is on `order_items`.
- If an order has items from multiple sellers, allow it to count once for each seller and state this behavior.
- Aggregate seller item metrics before joining to payment rows if payment behavior is requested.
- Do not use `sellers` alone to infer performance; it contains location metadata only.

## Status And Null Rules

- Use delivered-only filtering for fulfilled seller performance when requested.
- Include all statuses for placed seller order volume unless the user asks for completed performance.
- Exclude null `order_items.seller_id` from seller rankings.
- Exclude null `order_items.price` from seller sales sums.
- Exclude null `order_items.freight_value` from freight sums.
- Require non-null `order_items.shipping_limit_date` and `orders.order_delivered_carrier_date` for shipping-deadline miss metrics.
- Require non-null actual and estimated delivery dates for customer late-delivery metrics.
- Do not treat missing review rows as zero review score.
- Do not drop sellers with no review when analyzing sales unless review metrics are the focus.
- Keep null seller city/state as unknown if seller geography is requested.

## Clarify When

- The user says "best sellers" without defining sales, delivery, reviews, or volume.
- The user says "bad sellers" without defining late rate, low score, cancellations, or another metric.
- The user asks for seller performance but does not specify whether multi-seller orders should count for each seller.
- The user asks for shipping performance but does not specify deadline miss or customer delivery lateness.
- Ambiguous query example: "Who are the best sellers?"
- Ambiguous query example: "Find sellers with poor performance."
- Ambiguous query example: "Compare seller shipping quality."

## Prohibited Interpretations

- Do not infer seller profit, commission, or costs.
- Do not use `sellers` table row count as seller sales activity.
- Do not count item rows as seller order count.
- Do not use payment value as seller revenue without careful order-level allocation; payment rows are order-level, not seller-level.
- Do not assign all payment value to every seller in a multi-seller order.
- Do not infer inventory, stockouts, or fulfillment capacity.
- Do not infer carrier performance from seller data.
- Do not treat missing reviews as negative reviews.
- Do not use customer geography as seller geography.
- Do not use `customer_id` as seller or buyer-stable identity.

## Unsupported Requests

- Seller profitability, commission, payout, or cost analysis.
- Inventory or stock availability by seller.
- Carrier-specific seller shipping performance.
- Seller account tenure or onboarding analysis.
- Payment allocation by seller in multi-seller orders unless a clear allocation rule is supplied.

## Limitations

- Seller ownership is item-level; order-level metrics can duplicate across sellers in multi-seller orders.
- The schema has seller location but not seller profile, tenure, inventory, or commission fields.
- Reviews are order-level and may not identify which seller caused satisfaction issues in multi-seller orders.
- Payment records are order-level and not directly allocated to sellers.
- Shipping-limit metrics depend on non-null carrier handoff timestamps.

## Positive Examples

- "Rank sellers by delivered merchandise value."
- "Show item-line count and distinct order count by seller_state."
- "Calculate seller shipping deadline miss rate."
- "Compare average review score by seller_id."
- "Find sellers with high sales and high late delivery rate."

## Negative Examples

- "Calculate seller profit."
- "Assign full payment_value to each seller in an order."
- "Use sellers table count as active sales count."
- "Treat missing reviews as one-star reviews."
- "Rank carriers by seller performance."
