---
document_id: customer-value
doc_type: analysis_query_rule
query_type: customer_value
title: Olist Customer Value Query Rules
language: en
version: "1.0"
business_entities: [customers, orders, order_items, payments]
source_tables: [customers, orders, order_items, order_payments]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Customer Value Query Rules

## Definition

Customer value analysis measures customer-level monetary contribution and order behavior using `customers.customer_unique_id` as the customer grain. Default monetary value comes from item price, with freight or payment value included only when requested.

## Supported Intents

- Rank customers by total merchandise value.
- Calculate average order value per customer.
- Segment customers by monetary value and purchase count.
- Compute simple RFM-style recency, frequency, and monetary metrics from available fields.
- Compare customer value by customer state or city.
- Identify high-value repeat customers.
- Compare value by category or seller only when the join grain is controlled.

## Default Metrics

- `SUM(order_items.price)` per `customers.customer_unique_id` for merchandise value.
- `SUM(order_items.freight_value)` only when freight spend is requested.
- `SUM(order_items.price + order_items.freight_value)` only when total paid-like item value including freight is requested.
- `COUNT(DISTINCT orders.order_id)` for purchase frequency.
- `SUM(order_items.price) / COUNT(DISTINCT orders.order_id)` for average merchandise order value.
- `MAX(orders.order_purchase_timestamp)` for last observed purchase.
- Recency as date difference from a specified analysis date to last purchase.

## Entity Grain

- Default grain: `customers.customer_unique_id`.
- Order grain: `orders.order_id`.
- Item grain for monetary aggregation before customer rollup: `order_items.order_id`, `order_items.order_item_id`.
- Region grain, when requested: `customers.customer_state` or `customers.customer_city`.

## Time Basis

- Use `orders.order_purchase_timestamp` for purchase recency, cohorts, and period filters.
- Use delivered date only when the user asks for value of delivered orders by delivery period.
- Use payment records only for payment-value questions, not for default customer value.
- Require a reference date for recency unless the query defines one or the implementation uses max observed purchase timestamp explicitly.

## Required Tables

- `customers`
- `orders`
- `order_items`
- `order_payments` only when the user asks for paid amount or payment method value.
- `products` only when value by product/category is requested.
- `product_category_name_translation` only when English category names are requested.

## Join Constraints

- Join `orders` to `customers` on `orders.customer_id = customers.customer_id`.
- Join `orders` to `order_items` on `orders.order_id = order_items.order_id`.
- Aggregate item monetary fields to `order_id` before joining to `order_payments`.
- Use `customers.customer_unique_id` as the customer grouping key.
- Count orders with `COUNT(DISTINCT orders.order_id)` after joining to items.
- For category value, aggregate at customer-category level and state that multi-category orders contribute to each purchased category.
- For seller value, aggregate at customer-seller level and state that one customer can appear under many sellers.
- Do not join reviews unless satisfaction is explicitly part of the value question.
- Do not join geolocation unless coordinates or zip-prefix mapping is explicitly needed.
- Do not use `customer_id` for lifetime or repeat customer value.

## Status And Null Rules

- Filter to `orders.order_status = 'delivered'` when the user asks for realized, completed, fulfilled, or delivered customer value.
- Include all statuses for placed-order value unless the user requests completed value.
- Exclude rows with null `customers.customer_unique_id`.
- Exclude null `order_items.price` from merchandise value sums.
- Exclude null `order_items.freight_value` from freight value sums.
- Do not coalesce null price or freight to zero unless the query explicitly asks for missing-value handling.
- Keep customers with one order in value ranking unless repeat-only is requested.
- Exclude null `order_purchase_timestamp` from recency and cohort metrics.
- Do not treat missing review rows as zero satisfaction.
- Do not treat missing payment rows as zero value unless payment completeness is being audited.

## Clarify When

- The user says "customer value" but does not specify merchandise, freight-included, or payment value.
- The user asks for RFM without a recency reference date or scoring bins.
- The user asks for "best customers" without defining best by spend, frequency, recency, or satisfaction.
- The user asks for lifetime value; clarify that only observed historical value is available.
- Ambiguous query example: "Find our most valuable customers."
- Ambiguous query example: "Create RFM groups."
- Ambiguous query example: "Show high-value customers by category."

## Prohibited Interpretations

- Do not infer profit, margin, acquisition cost, or true lifetime value.
- Do not use `order_payments.payment_value` as default value when item price is sufficient.
- Do not add payment value and item price together.
- Do not group by `customers.customer_id` for customer value.
- Do not use order count alone as customer value unless the user asks for frequency.
- Do not use review score as monetary value.
- Do not infer future value or predictive CLV from this rule.
- Do not assume all statuses are revenue-realized.
- Do not count duplicate item-payment joins in customer spend.
- Do not use product dimensions as monetary value proxies.

## Unsupported Requests

- Profit-based CLV, margin-based customer value, or customer acquisition cost.
- Predictive lifetime value models without extra modeling instructions and labels.
- Demographic value segmentation by age, gender, income, or household.
- Refund-adjusted or tax-adjusted customer value.
- Attribution of customer value to marketing channels.

## Limitations

- The schema supports observed order value, not true lifetime value.
- Monetary totals can be inflated if item and payment rows are joined without pre-aggregation.
- `customer_unique_id` is the best available customer key but may not represent household identity.
- Recency requires a reference date decision.
- Category and seller splits can duplicate a customer's total across multiple dimensions.

## Positive Examples

- "Rank customer_unique_id by delivered merchandise spend."
- "Calculate average merchandise order value per customer."
- "Create RFM metrics using max purchase date as the reference date."
- "Compare high-value customer counts by customer_state."
- "Find repeat customers with merchandise spend over 1000."

## Negative Examples

- "Calculate customer profit."
- "Predict future CLV."
- "Group customers by customer_id."
- "Add payment_value and item price for total customer value."
- "Segment value by customer age."
