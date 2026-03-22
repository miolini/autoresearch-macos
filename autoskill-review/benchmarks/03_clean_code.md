# Benchmark: Clean Code (Don't Over-Criticize)

## Input Diff

```diff
diff --git a/src/utils/format.js b/src/utils/format.js
+export function formatCurrency(amount, currency = 'USD') {
+  return new Intl.NumberFormat('en-US', {
+    style: 'currency',
+    currency,
+  }).format(amount);
+}
```

## Expected Behaviors

The review MUST:
1. Recognize this is clean, correct code
2. NOT invent problems that don't exist
3. NOT suggest unnecessary rewrites
4. May note minor things (locale could be parameterized) as nitpicks only
5. Overall tone should be approving

## Scoring

- 1.0: Approves with minimal or no nitpicks
- 0.5: Approves but with unnecessary/pedantic suggestions
- 0.0: Blocks or invents serious issues that don't exist
