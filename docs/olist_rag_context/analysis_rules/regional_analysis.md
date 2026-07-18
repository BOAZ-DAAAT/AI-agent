---
document_id: regional-analysis
doc_type: analysis_query_rule
query_type: regional_analysis
title: Olist Regional Analysis Query Rules
language: en
version: "1.0"
business_entities: [customers, sellers, geolocation, orders]
source_tables: [customers, sellers, geolocation, orders, order_items, order_payments, order_reviews, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Regional Analysis Query Rules

## Definition

Regional analysis groups Olist orders, customers, sellers, delivery, payments, or reviews by customer, seller, or geolocation region fields. Customer geography and seller geography are distinct and must not be mixed without explicit intent.

## Supported Intents

- Compare orders, sales, customers, reviews, delivery delay, or payments by customer state/city.
- Compare seller activity or performance by seller state/city.
- Compare customer region to seller region.
- Map zip-prefix coordinates when latitude/longitude is explicitly requested.
- Analyze regional category mix.
- Compare delivery delay by origin seller region and destination customer region.

## Default Metrics

- `COUNT(DISTINCT orders.order_id)` for regional order count.
- `COUNT(DISTINCT customers.customer_unique_id)` for regional customer count.
- `SUM(order_items.price)` for regional merchandise value.
- `AVG(order_reviews.review_score)` for regional satisfaction when requested.
- Late delivery rate for regional delivery analysis.
- `COUNT(DISTINCT order_items.seller_id)` for active seller count by seller region.

## Entity Grain

- Customer region grain: `customers.customer_state`, `customers.customer_city`, or `customers.customer_zip_code_prefix`.
- Seller region grain: `sellers.seller_state`, `sellers.seller_city`, or `sellers.seller_zip_code_prefix`.
- Geolocation grain: `geolocation.geolocation_zip_code_prefix`.
- Order grain: `orders.order_id`.
- Customer grain: `customers.customer_unique_id`.
- Seller grain: `order_items.seller_id`.

## Time Basis

- Use `orders.order_purchase_timestamp` as the default time basis for regional order and sales trends.
- Use delivered date only when the query asks for delivery-period regional analysis.
- Use review creation date only for regional review trends.
- Use no time basis for static customer/seller location distribution unless a period is requested.

## Required Tables

- `customers` for customer region analysis.
- `orders` for order, time, status, delivery, or customer-region order metrics.
- `order_items` for sales, seller, product, freight, or category metrics.
- `sellers` for seller region analysis.
- `geolocation` only when coordinates or zip-prefix location mapping is requested.
- `order_reviews` only when satisfaction by region is requested.
- `order_payments` only when payment behavior by region is requested.
- `products` and `product_category_name_translation` only when category by region is requested.

## Join Constraints

- Join customer geography through `orders.customer_id = customers.customer_id`.
- Join seller geography through `order_items.seller_id = sellers.seller_id`.
- Join customer zip to geolocation on `customers.customer_zip_code_prefix = geolocation.geolocation_zip_code_prefix`.
- Join seller zip to geolocation on `sellers.seller_zip_code_prefix = geolocation.geolocation_zip_code_prefix`.
- Use customer region for buyer/destination questions.
- Use seller region for seller/origin questions.
- Count orders with `COUNT(DISTINCT orders.order_id)` after joining to item, seller, payment, review, or geolocation tables.
- Aggregate geolocation by zip prefix before joining if duplicate geolocation rows exist.
- Do not join `customers.customer_city` to `geolocation.geolocation_city`; use zip-prefix joins for coordinates.
- Do not compare customer and seller states without clearly labeling destination versus origin.

## Status And Null Rules

- Use all order statuses for regional demand/order-intake analysis unless the user asks for completed orders.
- Filter to delivered orders for completed regional sales or delivery metrics.
- Keep null city/state as unknown when reporting all regions.
- Exclude null zip prefixes from coordinate joins.
- Do not drop orders without geolocation rows unless mapping coordinates is the required output.
- Exclude null `order_purchase_timestamp` from time trends.
- Exclude null `order_items.price` from regional sales sums.
- Do not treat seller state as customer state.
- Do not treat customer zip prefix as exact street address.
- Do not infer regions outside the state/city/zip-prefix fields in the schema.

## Clarify When

- The user says "region" without specifying customer/destination region or seller/origin region.
- The user asks for "map" or "distance" without saying whether to use customer zip, seller zip, or both.
- The user asks for regional sales without saying whether sales means merchandise value or payment value.
- The user asks for state-level or city-level output but includes zip-prefix examples.
- Ambiguous query example: "Show sales by region."
- Ambiguous query example: "Map delivery delays."
- Ambiguous query example: "Compare north and south performance."

## Prohibited Interpretations

- Do not mix customer and seller geography under a generic region label.
- Do not infer exact address from zip prefix.
- Do not use city-name joins for coordinates.
- Do not assume geolocation rows are unique per zip prefix.
- Do not count geolocation rows as customers, sellers, or orders.
- Do not infer official macro-regions like North/South unless a mapping table or explicit rule is supplied.
- Do not use seller location for customer demand unless requested.
- Do not use customer location for seller supply unless requested.
- Do not infer distance or route time without explicit geolocation calculation and caveats.
- Do not use `customer_id` as the customer grain for regional customer counts.

## Unsupported Requests

- Exact addresses, street-level maps, or route optimization.
- Official macro-region grouping without a supplied mapping.
- Population-normalized regional penetration; population data is absent.
- Distance-to-warehouse analysis; warehouse data is absent.
- International region analysis outside the loaded Olist geography.

## Limitations

- Geography is available as city/state and zip prefix, not full address.
- Geolocation is approximate and keyed by zip prefix.
- Duplicate geolocation rows may require aggregation before coordinate use.
- Customer and seller region answer different business questions.
- Regional customer counts must use `customer_unique_id`, not `customer_id`.

## Positive Examples

- "Show delivered merchandise sales by customer_state."
- "Count active sellers by seller_state."
- "Compare average delivery delay by customer city."
- "Map customer zip-prefix centroids for orders in SP."
- "Compare customer_state and seller_state pairs by order count."

## Negative Examples

- "Use seller_state as buyer region."
- "Join city names to geolocation for coordinates."
- "Infer Brazilian macro-regions without a mapping table."
- "Count geolocation rows as customers."
- "Calculate street-level delivery distance."
