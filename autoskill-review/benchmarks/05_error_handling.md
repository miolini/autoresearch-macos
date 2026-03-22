# Benchmark: Missing Error Handling

## Input Diff

```diff
diff --git a/src/services/payment.js b/src/services/payment.js
+async function processPayment(orderId, amount) {
+  const order = await db.orders.findById(orderId);
+  const result = await paymentGateway.charge(order.customerId, amount);
+  await db.orders.update(orderId, { status: 'paid', transactionId: result.id });
+  return result;
+}
```

## Expected Behaviors

The review MUST:
1. Note missing error handling for critical payment flow
2. Ask: what if order not found? what if charge fails?
3. Mark as blocker (payment code needs error handling)
4. Suggest try/catch or validation checks
5. Mention potential for orphaned states (charge succeeds but update fails)

## Scoring

- 1.0: Identifies missing error handling with appropriate severity
- 0.5: Mentions one issue but misses the critical ones
- 0.0: Approves without noting error handling gaps
