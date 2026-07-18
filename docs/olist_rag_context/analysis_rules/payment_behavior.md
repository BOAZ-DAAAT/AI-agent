---
document_id: payment-behavior
doc_type: analysis_query_rule
query_type: payment_behavior
title: Olist Payment Behavior Query Rules
language: en
version: "1.0"
business_entities: [payments, orders, customers]
source_tables: [order_payments, orders, customers, order_items, products]
grounding_level: schema_grounded
intended_use: SQL planning rule retrieval
prohibited_use: live metric evidence
source_schema: DATA_Analyst_Assistant_Agent/agents/sql/data/db_schema.json
---

# Payment Behavior Query Rules

## Definition

Payment behavior analysis measures payment method, installment use, payment row count, and payment value from `order_payments`, joined to orders and customer/item context only when needed.

## Supported Intents

- Compare payment method distribution.
- Analyze installment counts and installment usage.
- Calculate payment value by method.
- Compare payment behavior by customer region, order status, category, or time.
- Identify orders with multiple payment records.
- Compare average payment value by payment type.
- Analyze credit card versus boleto behavior.

## Default Metrics

- `COUNT(*)` from `order_payments` for payment record count.
- `COUNT(DISTINCT order_payments.order_id)` for paid order count.
- `SUM(order_payments.payment_value)` for total payment value.
- `AVG(order_payments.payment_value)` for average payment record value.
- `AVG(order_payments.payment_installments)` for average installments.
- `COUNT(DISTINCT order_id) FILTER (WHERE payment_installments > 1)` for installment order count after order-level aggregation.
- Payment method share by `payment_type`.

## Entity Grain

- Default grain: payment record, identified by `order_id` and `payment_sequential`.
- Order grain: `order_payments.order_id`.
- Customer grain, when requested: `customers.customer_unique_id`.
- Category or seller grain requires item joins and explicit allocation caution.

## Time Basis

- Use `orders.order_purchase_timestamp` as the default time basis for payment behavior trends.
- No payment timestamp exists in `order_payments`; do not invent one.
- Use approval time only when asking about approved orders.
- Use delivery time only when payment behavior is compared with delivery outcomes.

## Required Tables

- `order_payments`
- `orders` when order status, purchase time, approval, delivery, or customer joins are needed.
- `customers` only when customer identity or region is requested.
- `order_items` only when category, seller, price, freight, or item context is requested.
- `products` only when product/category payment behavior is requested.
- `product_category_name_translation` only when English category names are requested.

## Join Constraints

- Join `order_payments` to `orders` on `order_payments.order_id = orders.order_id`.
- Join `orders` to `customers` on `orders.customer_id = customers.customer_id`.
- Aggregate payment rows to `order_id` before joining to `order_items` for item/category/seller analysis.
- Count paid orders with `COUNT(DISTINCT order_payments.order_id)`.
- Count payment records with `COUNT(*)` only when payment splits or records are requested.
- For payment method mix, be explicit whether the denominator is payment records or distinct orders.
- For orders with multiple payment methods, decide whether to count each method record or classify the order as mixed.
- Do not join payments directly to item rows and then sum `payment_value`; this multiplies payment totals.
- Do not allocate order payment value to sellers or categories without a stated allocation rule.
- Do not add `payment_value` to `order_items.price`; they are different monetary views.

## Status And Null Rules

- Include all statuses for payment-method distribution unless the user asks for delivered, completed, or approved orders.
- Filter to delivered orders when payment behavior of completed orders is requested.
- Exclude null `payment_type` from payment method grouping unless an unknown bucket is requested.
- Exclude null `payment_value` from payment value sums and averages.
- Exclude null `payment_installments` from installment averages.
- Treat `payment_installments = 1` as single-payment, not missing.
- Treat `payment_sequential` as payment record order, not number of installments.
- Keep zero-value payment rows only if they exist and the query asks for payment records; otherwise flag them as data-quality cases.
- Do not infer payment date from purchase, approval, or delivery timestamps.
- Do not treat missing payment rows as unpaid orders unless the query is explicitly auditing missing payments.

## Clarify When

- The user says "payment share" without specifying payment-record share or order share.
- The user asks for "revenue by payment type" without confirming payment value versus item sales allocation.
- The user asks for category or seller payment value without an allocation rule.
- The user asks for "installment customers" without defining any installment threshold beyond `payment_installments > 1`.
- Ambiguous query example: "Show payment mix."
- Ambiguous query example: "Revenue by payment method and seller."
- Ambiguous query example: "Which customers use installments?"

## Prohibited Interpretations

- Do not treat `payment_sequential` as installment count.
- Do not treat `payment_installments` as number of payment rows.
- Do not infer payment date; no payment timestamp exists.
- Do not sum payment value after a direct many-to-many join with item rows.
- Do not allocate full order payment to every seller or category in a multi-item order.
- Do not use item price as payment value unless the query changes to sales analysis.
- Do not infer failed, refunded, or chargeback payments; no such fields exist.
- Do not infer credit risk or customer income.
- Do not treat missing payment rows as zero unless auditing missing payments.
- Do not convert boleto or credit_card labels to broader payment families unless requested.

## Unsupported Requests

- Payment failure, fraud, chargeback, refund, or authorization timing analysis.
- Exact payment date or settlement date analysis.
- Credit risk, income, or bank account analysis.
- Seller payout or payment allocation without a rule.
- Tax, fee, or processor-cost analysis.

## Limitations

- `order_payments` has no payment timestamp.
- One order can have multiple payment rows.
- Payment value is order-level payment information and is not natively allocated to item, seller, or category.
- Payment method shares depend on whether the denominator is records or orders.
- Refunds, failures, fees, and chargebacks are not represented.

## Positive Examples

- "Show payment_type share by distinct order count."
- "Calculate total payment_value by payment_type."
- "Find orders with more than one payment_sequential row."
- "Compare average installments by customer_state."
- "Analyze delivered orders paid with credit_card versus boleto."

## Negative Examples

- "Use payment_sequential as installments."
- "Join payments to order_items and sum payment_value by category without allocation."
- "Find payment failures."
- "Trend payments by payment date."
- "Calculate processor fees."
