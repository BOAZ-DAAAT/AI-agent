---
document_id: purchase-frequency
doc_type: analysis_query_rule
query_type: purchase_frequency
title: Olist Purchase Frequency Query Rules
language: en
version: "1.0"
business_entities: [customers, orders]
source_tables: [customers, orders]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Purchase Frequency Query Rules

## Definition

Purchase frequency measures how often the same real customer purchases, using `customers.customer_unique_id` as the customer identity and distinct `orders.order_id` as the purchase count.

## Supported Intents

- Count repeat customers.
- Segment customers by number of distinct orders.
- Calculate average orders per customer.
- Compare one-time and repeat purchasers.
- Analyze reorder intervals using purchase timestamps.
- Rank customers by purchase count.
- Compare purchase frequency by region or category.

## Default Metrics

- `COUNT(DISTINCT orders.order_id)` per `customers.customer_unique_id` for customer purchase count.
- `COUNT(DISTINCT customers.customer_unique_id)` for customer count.
- `AVG(customer_order_count)` for average purchase frequency.
- `COUNT(*) FILTER (WHERE customer_order_count = 1)` for one-time customer count.
- `COUNT(*) FILTER (WHERE customer_order_count >= 2)` for repeat customer count.
- Date difference between consecutive `orders.order_purchase_timestamp` values for reorder interval.

## Entity Grain

- Default grain: `customers.customer_unique_id`.
- Order grain: `orders.order_id`.
- Region grain, when requested: `customers.customer_state` or `customers.customer_city`.
- Category grain, when requested: translated product category after joining through item and product tables.

## Time Basis

- Use `orders.order_purchase_timestamp` as the default purchase time.
- Use a customer's first purchase timestamp for first-order cohort questions.
- Use consecutive purchase timestamps for reorder interval questions.
- Do not use delivery, approval, review, or payment timestamps as the purchase time.

## Required Tables

- `customers`
- `orders`
- `order_items` only when frequency is sliced by product, category, seller, item value, or freight.
- `products` only when product category or product attributes are requested.
- `product_category_name_translation` only when English category names are requested.
- `order_payments` only when payment behavior is part of the question.

## Join Constraints

- Join `orders` to `customers` on `orders.customer_id = customers.customer_id`.
- Use `customers.customer_unique_id` for any customer-level grouping.
- Use `orders.customer_id` only as the join key to `customers`.
- Count purchases with `COUNT(DISTINCT orders.order_id)`.
- If joining `order_items`, preserve order counts with distinct `orders.order_id`.
- If joining `order_payments`, aggregate payments by `order_id` before customer-level frequency analysis.
- For category frequency, decide whether the grain is customer-category pairs or all customer orders before grouping.
- Do not group repeat customers by `customers.customer_id`; that fragments repeat behavior.
- Do not use `order_items.order_item_id` as a purchase count.
- Do not join `geolocation` unless coordinates or zip-prefix geography are explicitly requested.

## Status And Null Rules

- Include all order statuses for general purchase-attempt frequency unless the user asks for completed purchases.
- Filter to `orders.order_status = 'delivered'` for completed, delivered, or fulfilled purchase frequency.
- Exclude rows with null `customers.customer_unique_id`.
- Exclude rows with null `orders.order_id`.
- Exclude rows with null `orders.order_purchase_timestamp` when using time windows, cohorts, or reorder intervals.
- Do not require `order_delivered_customer_date` unless the user asks for delivered purchases.
- Keep customers with exactly one distinct order in the denominator for average purchase frequency unless repeat-only is requested.
- Treat multiple item rows in one order as one purchase.
- Treat multiple payment rows in one order as one purchase.
- Do not infer churn unless an inactivity threshold and observation window are provided.

## Clarify When

- The user says "frequent" without a numeric threshold or ranking requirement.
- The user asks for "loyal customers" without defining loyalty.
- The user asks for churn, retention, or active customers without an observation window.
- The user asks for repeat purchases by category without saying whether repeat means same category or any later order.
- Ambiguous query example: "Find frequent buyers."
- Ambiguous query example: "Who are loyal customers?"
- Ambiguous query example: "Analyze retention by product type."

## Prohibited Interpretations

- Do not use `customer_id` as customer identity for repeat purchase analysis.
- Do not use total spend as purchase frequency.
- Do not use item-line count as order count.
- Do not use payment-row count as purchase count.
- Do not treat multiple items in one order as multiple purchases unless the user explicitly asks for item purchases.
- Do not infer household, account, or demographic identity beyond `customer_unique_id`.
- Do not label a customer churned without a stated inactivity rule.
- Do not infer subscription behavior; no subscription field exists.
- Do not compute lifetime value unless monetary metrics are explicitly requested.
- Do not treat review count as purchase count.

## Unsupported Requests

- True retention rate requiring customer exposure outside the observed order table.
- Demographic purchase frequency by age, gender, income, or household.
- Subscription cadence, renewal, or cancellation analysis.
- Web/app visit frequency or browsing frequency.
- Churn prediction without a specified label rule.

## Limitations

- Olist `customer_unique_id` is the only stable customer key in this schema.
- The dataset only shows observed orders; it cannot prove a customer never purchased elsewhere.
- Sparse repeat purchasing can make reorder interval estimates unstable.
- Order status interpretation changes the numerator and must be explicit for completed-purchase questions.
- Product/category slices can duplicate customer-order membership across multi-item orders.

## Positive Examples

- "Count customers with two or more distinct orders."
- "Calculate average distinct orders per customer_unique_id."
- "Rank customer_unique_id values by delivered purchase count."
- "Show repeat customer rate by customer_state."
- "Measure days between first and second purchase."

## Negative Examples

- "Use customer_id to find repeat customers."
- "Rank loyal customers by total spend only."
- "Count order_items rows as purchases."
- "Find churned users without defining churn."
- "Analyze purchase frequency by age."
