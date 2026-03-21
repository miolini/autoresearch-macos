# Benchmark: Simple Function

## Input Code

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

## Expected Behaviors

The explanation MUST:
1. Identify this as Python
2. Mention "factorial" or "product of integers"
3. Identify the recursion pattern
4. Note the base case (n <= 1)
5. Mention potential stack overflow for large n

## Scoring

- 1.0: All 5 behaviors present
- 0.5: 3-4 behaviors present
- 0.0: 2 or fewer behaviors present
