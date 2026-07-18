---
document_id: product-category
doc_type: analysis_query_rule
query_type: product_category
title: Olist Product Category Query Rules
language: en
version: "1.0"
business_entities: [products, categories, orders, order_items]
source_tables: [products, product_category_name_translation, order_items, orders, order_reviews]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Product Category Query Rules

## Definition

Product category analysis groups ordered products by `products.product_category_name` or its English translation in `product_category_name_translation`, then measures order, item, sales, freight, delivery, or review metrics by category.

## Supported Intents

- Rank categories by order count, item count, or merchandise value.
- Compare category-level average price, freight, review score, or delivery delay.
- Translate Portuguese category names to English for reporting.
- Identify categories with missing category names.
- Analyze category performance over time.
- Compare category mix by customer or seller region.

## Default Metrics

- `COUNT(DISTINCT orders.order_id)` for category order count.
- `COUNT(*)` from `order_items` for category item-line count.
- `SUM(order_items.price)` for category merchandise value.
- `AVG(order_items.price)` for average item price.
- `SUM(order_items.freight_value)` for category freight.
- `AVG(order_reviews.review_score)` only when satisfaction is requested.
- Late rate only when delivery delay is requested.

## Entity Grain

- Default grain: category.
- Category key: `products.product_category_name`.
- English category label: `product_category_name_translation.product_category_name_english`.
- Item grain: `order_items.order_id`, `order_items.order_item_id`.
- Order grain: `orders.order_id`.

## Time Basis

- Use `orders.order_purchase_timestamp` as the default time basis for category sales and order trends.
- Use `orders.order_delivered_customer_date` only for delivered-date category questions.
- Use `order_reviews.review_creation_date` only for category review timing.
- Use `order_items.shipping_limit_date` only for category shipping-deadline questions.

## Required Tables

- `order_items`
- `products`
- `orders` when order count, time, status, customer region, or delivery metrics are needed.
- `product_category_name_translation` when English category names are needed.
- `order_reviews` only when review satisfaction is requested.
- `customers` only when customer region is requested.
- `sellers` only when seller region is requested.

## Join Constraints

- Join `order_items` to `products` on `order_items.product_id = products.product_id`.
- Join `products` to `product_category_name_translation` on `products.product_category_name = product_category_name_translation.product_category_name`.
- Join `order_items` to `orders` on `order_items.order_id = orders.order_id`.
- Join `orders` to `order_reviews` on `orders.order_id = order_reviews.order_id` only for review metrics.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id` only for customer region.
- Join `order_items` to `sellers` on `order_items.seller_id = sellers.seller_id` only for seller region or seller analysis.
- Count category orders with `COUNT(DISTINCT orders.order_id)`, not item rows.
- Use item rows for item-line count and monetary sums because category is assigned at product/item grain.
- If one order contains multiple categories, allow that order to count once in each category and state this behavior.
- Do not aggregate category metrics by `product_id` unless product-level output is requested.

## Status And Null Rules

- Use delivered-order filtering only when the user asks for completed, delivered, or fulfilled category performance.
- Keep null `products.product_category_name` as unknown when reporting all category distribution.
- Keep null English translations as untranslated categories; do not drop the product by default.
- Exclude null `order_items.price` from price sums and averages.
- Exclude null `order_items.freight_value` from freight sums and averages.
- Exclude null `orders.order_purchase_timestamp` from category trend grouping.
- Do not require reviews for category sales/order metrics.
- Do not treat missing review rows as zero review score.
- Do not treat missing product dimensions as missing category.
- Preserve Portuguese category name if English translation is unavailable.

## Clarify When

- The user asks for "top categories" without saying top by orders, items, sales, reviews, or delivery.
- The user asks for category names without specifying Portuguese source names or English translations.
- The user asks for category performance without defining the performance metric.
- The user asks for category share and an order has multiple categories.
- Ambiguous query example: "What are the best categories?"
- Ambiguous query example: "Show category performance."
- Ambiguous query example: "Compare categories by popularity."

## Prohibited Interpretations

- Do not use product dimensions as category labels.
- Do not treat `product_category_name_english` as the join key to products.
- Do not count product master rows as sold products.
- Do not count category order volume with plain `COUNT(*)` after item joins unless item-line count is requested.
- Do not infer human-readable category names outside the translation table.
- Do not drop unknown categories unless the user asks to exclude them.
- Do not infer product quantity beyond item rows.
- Do not use review comments as category taxonomy.
- Do not infer brand, department, or hierarchy levels; the schema has one category field.
- Do not assume categories are mutually exclusive at order level.

## Unsupported Requests

- Category hierarchy, department tree, or taxonomy levels beyond the single category column.
- Brand-level analysis unless brand is encoded outside this schema.
- Product text search by actual name or description; only name/description length fields exist.
- Inventory by category.
- Category margin or profitability.

## Limitations

- Product category is stored on product master records, not directly on orders.
- English labels depend on translation-table coverage.
- Multi-category orders can make category order counts sum above total distinct orders.
- Product names and descriptions are not present, only length fields.
- Category metrics inherit status and null limitations from joined order/item/review tables.

## Positive Examples

- "Rank English product categories by merchandise sales."
- "Count distinct delivered orders per category."
- "Show item-line count and average price by Portuguese category."
- "Compare average review score by category."
- "Trend category sales by purchase month."

## Negative Examples

- "Build a three-level category hierarchy."
- "Analyze brand performance by category."
- "Use product master count as sold item count."
- "Treat unknown category as zero sales."
- "Join category using product_category_name_english to products."
