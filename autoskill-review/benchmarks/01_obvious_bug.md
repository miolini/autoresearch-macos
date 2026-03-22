# Benchmark: Obvious Bug

## Input Diff

```diff
diff --git a/src/utils/array.js b/src/utils/array.js
+function findMax(numbers) {
+  let max = 0;
+  for (const n of numbers) {
+    if (n > max) {
+      max = n;
+    }
+  }
+  return max;
+}
```

## Expected Behaviors

The review MUST:
1. Identify the bug: initializing max to 0 fails for all-negative arrays
2. Mark it as a blocker or high severity
3. Suggest a fix (use -Infinity, or numbers[0], or Math.max)
4. Be specific about the failure case
5. NOT just say "looks good" or approve without comment

## Scoring

- 1.0: Catches the bug with correct severity and fix
- 0.5: Mentions something is off but wrong diagnosis
- 0.0: Approves the code or misses the bug entirely
