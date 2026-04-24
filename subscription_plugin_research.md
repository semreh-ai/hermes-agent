# WooCommerce Subscription Plugin Research
## For caferico.be

## 1. WooCommerce Subscriptions (Official Automattic Plugin)

### Latest Version Compatibility
- WordPress 6.9.x: Supported. WooCommerce.com runs WP 6.9.4.
- WooCommerce 10.7.x: Supported. WooCommerce.com runs WC 10.7.0-rc.1.
- Source: https://woocommerce.com/products/woocommerce-subscriptions/

### Pricing
- Annual: ~EUR 244 / $239 USD per year
- Hidden costs: "Buy Once or Subscribe" add-on (~$49/year) for mixed carts.

### Product Support
- Simple products: Yes
- Variable products: Yes
- Mixed carts: Limited natively; add-on recommended.

### Store Operations
- Shipping: Yes
- Taxes: Yes
- Stock: Yes
- Coupons: Yes (recurring, sign-up fee, limited-use)

### Customer Self-Service
- Pause: Yes
- Cancel: Yes
- Skip: Yes
- Change payment: Yes (but see Mollie caveat)

### Failed Payment Retry Logic
- Built-in retry system with customizable rules.
- Source: https://woocommerce.com/document/subscriptions/failed-payment-retry/

### Renewal Order Model
- Standard WC orders. Compatible with order hooks (important for Rico Order Alerts).

### Divi Theme Compatibility
- Yes. No known conflicts.

### Extra Paid Add-ons Required?
- "Buy Once or Subscribe" for mixed carts (~$49/year).

### Mollie Payments for WooCommerce Issues
- Known issues:
  - GitHub issue #571: Customers cannot easily change payment method for Mollie recurring payments.
  - If card declines, mandate expires; customer must re-purchase.
  - SEPA Direct Debit is the reliable recurring method.
- Source: https://github.com/mollie/WooCommerce/issues/571

---

## 2. YITH WooCommerce Subscription

### Latest Version Compatibility
- WordPress 6.9.x: Supported.
- WooCommerce 10.7.x: Supported (updated for WC 10.4+).
- Source: https://yithemes.com/latest-updates/

### Pricing
- Annual: $199.99 USD (~EUR 199.99) per year
- Hidden costs: Many features require Premium.

### Product Support
- Simple products: Yes (Free & Premium)
- Variable products: Yes (Premium)
- Mixed carts: Limited.

### Store Operations
- Shipping: Yes (Premium)
- Taxes: Yes
- Stock: Yes
- Coupons: Yes (Premium)

### Customer Self-Service
- Pause: Yes (Premium)
- Cancel: Yes
- Skip: Not explicitly mentioned
- Change payment: Unclear; Mollie not listed
- Upgrade/Downgrade: Yes (Premium)

### Failed Payment Retry Logic
- After 3 failed attempts, auto-cancel.
- "Renew Now" button for manual retry.
- No advanced dunning schedule.

### Renewal Order Model
- Custom model (separate YITH subscription table).
- May require testing with Rico Order Alerts.

### Divi Theme Compatibility
- General compatibility claimed, but not specifically tested with Divi.

### Extra Paid Add-ons Required?
- Core features in Premium; no major add-ons for standard subscription.

### Mollie Payments for WooCommerce Issues
- Not officially supported.
- Premium only lists Stripe & PayPal for automatic renewals.
- Free version only PayPal Standard.
- Impact: Likely dealbreaker for caferico.be.

---

## Verdict

**Recommendation: WooCommerce Subscriptions (Official Automattic Plugin)**

Rationale:
1. Mollie compatibility is non-negotiable. YITH does not support Mollie automatic renewals.
2. Standard WC orders ensure Rico Order Alerts works without changes.
3. Built-in retry system reduces churn.
4. Highest confidence with Divi.

Before going live:
- Test Mollie change-payment-method flow on staging.
- Consider enabling SEPA Direct Debit as primary recurring method.
- Budget for "Buy Once or Subscribe" add-on if selling one-time + subscription together.
- Configure failed-payment retry rules.

