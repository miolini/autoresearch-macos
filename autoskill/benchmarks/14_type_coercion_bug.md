# Benchmark: JavaScript Type Coercion Bug

## Input Code

```javascript
function isAdult(age) {
  if (age == '18') {
    return true;
  }
  if (age > 18) {
    return true;
  }
  return false;
}

// Bug: isAdult('19') returns false!
```

## Expected Behaviors

The explanation MUST:
1. Identify this as JavaScript
2. Explain it checks if someone is 18 or older
3. **CRITICAL**: Identify the type coercion bug with string comparison
4. Explain why '19' > 18 is false (string vs number comparison)
5. Note the inconsistent use of == vs > with mixed types

## Scoring

- 1.0: All 5 behaviors present (MUST include #3)
- 0.5: Explains the function but misses the bug
- 0.0: Claims the function works correctly
