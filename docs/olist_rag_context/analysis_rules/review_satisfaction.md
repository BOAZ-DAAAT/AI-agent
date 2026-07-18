---
document_id: review-satisfaction
doc_type: analysis_query_rule
query_type: review_satisfaction
title: Olist Review Satisfaction Query Rules
language: en
version: "1.0"
business_entities: [reviews, orders, customers, sellers, products]
source_tables: [order_reviews, orders, customers, order_items, sellers, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Review Satisfaction Query Rules

## Definition

Review satisfaction analysis uses `order_reviews.review_score` and optional review text/timestamps to measure customer satisfaction at review/order grain, then joins to orders, customers, products, sellers, or delivery fields only as needed.

## Supported Intents

- Calculate average review score.
- Count positive, neutral, or negative reviews when thresholds are specified or defaulted.
- Compare satisfaction by category, seller, customer region, delivery delay, or payment type.
- Analyze review response timing from review creation to answer timestamp.
- Count reviews with comments.
- Identify low-scoring orders for operational analysis.

## Default Metrics

- `AVG(order_reviews.review_score)` for average satisfaction.
- `COUNT(DISTINCT order_reviews.review_id)` for review count.
- Default positive reviews: `review_score >= 4`.
- Default negative reviews: `review_score <= 2`.
- Default neutral reviews: `review_score = 3`.
- Comment presence rate from non-null, non-empty `review_comment_message`.
- Review response time from `review_creation_date` to `review_answer_timestamp`.

## Entity Grain

- Default grain: `order_reviews.review_id`.
- Order grain: `order_reviews.order_id` / `orders.order_id`.
- Seller grain, when requested: `order_items.seller_id`.
- Category grain, when requested: translated product category through `order_items` and `products`.
- Customer grain, when requested: `customers.customer_unique_id`.

## Time Basis

- Use `order_reviews.review_creation_date` as the default review time.
- Use `order_reviews.review_answer_timestamp` only for response timing.
- Use `orders.order_purchase_timestamp` only when satisfaction is grouped by purchase period.
- Use `orders.order_delivered_customer_date` only when connecting satisfaction to delivery completion or delay.

## Required Tables

- `order_reviews`
- `orders` when order status, purchase time, delivery, customer, or item joins are needed.
- `order_items` when seller, product, category, item value, or freight context is needed.
- `products` only when product/category context is needed.
- `product_category_name_translation` only when English category names are requested.
- `customers` only when customer identity or region is requested.
- `order_payments` only when payment behavior is requested.

## Join Constraints

- Join `order_reviews` to `orders` on `order_reviews.order_id = orders.order_id`.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id` for customer context.
- Join `orders` to `order_items` on `orders.order_id = order_items.order_id` for seller/category/item context.
- Join `order_items` to `products` on `order_items.product_id = products.product_id`.
- Join translations through `products.product_category_name`.
- Use `COUNT(DISTINCT order_reviews.review_id)` after joining to item-level tables.
- If reviewing by seller or category, state that order-level reviews can be duplicated across multiple sellers/categories in the same order.
- Aggregate item-level context before combining with review counts when both sales and reviews are requested.
- Do not join payments unless payment type or payment amount is part of the satisfaction question.
- Do not use review text as structured reason categories unless the user requests text analysis and accepts qualitative limitations.

## Status And Null Rules

- Exclude null `review_score` from score averages.
- Keep reviews with null title or message in score metrics.
- Treat empty or null `review_comment_message` as no comment for comment-presence metrics.
- Require non-null `review_creation_date` for review trend analysis.
- Require non-null `review_answer_timestamp` for response-time metrics.
- Do not treat missing answer timestamp as zero response time.
- Do not require delivered status unless satisfaction must be limited to delivered orders.
- Do not treat missing review rows as zero satisfaction for orders.
- Use delivered-date fields only when satisfaction is analyzed against delivery delay.
- Keep low, neutral, and high score thresholds explicit when the user provides them.

## Clarify When

- The user says "satisfied" without defining score thresholds and defaults may change the result.
- The user asks for "review quality" without saying score, comment presence, or response time.
- The user asks for reasons for bad reviews but does not request text analysis.
- The user asks for seller satisfaction where orders may contain multiple sellers.
- Ambiguous query example: "Which sellers have unhappy customers?"
- Ambiguous query example: "Analyze review quality."
- Ambiguous query example: "Why are customers dissatisfied?"

## Prohibited Interpretations

- Do not infer sentiment beyond `review_score` unless text analysis is explicitly requested.
- Do not treat null review comments as negative comments.
- Do not treat missing reviews as zero-star reviews.
- Do not count item rows as review count.
- Do not assign an order-level review to one seller in a multi-seller order without stating the duplication rule.
- Do not use `review_creation_date` as purchase date.
- Do not infer NPS; no NPS field exists.
- Do not infer customer service agent identity; no such field exists.
- Do not translate or categorize free-text comments unless explicitly requested.
- Do not assume review score scale outside the observed 1-5 meaning in the schema.

## Unsupported Requests

- Net Promoter Score from absent NPS survey data.
- Verified textual topic modeling unless an explicit text-analysis task is requested.
- Customer service agent performance.
- Product defect classification without text modeling assumptions.
- Sentiment analysis in languages or categories not represented by structured fields.

## Limitations

- Reviews are order-level, not item-level or seller-specific.
- Multi-item or multi-seller orders can duplicate review attribution in joined outputs.
- Comment title and message are optional and often missing.
- Review response timestamp availability controls response-time denominators.
- The schema does not contain moderation status, helpfulness votes, or NPS data.

## Positive Examples

- "Calculate average review_score by purchase month."
- "Compare negative review rate by seller_id using review_score <= 2."
- "Show average review score by English category."
- "Measure review answer time for reviews with answer timestamps."
- "Compare late delivery rate and average review score."

## Negative Examples

- "Treat missing reviews as zero."
- "Calculate NPS."
- "Use review date as order purchase date."
- "Assign each review to only the first seller in an order."
- "Infer exact defect reasons without text analysis."
