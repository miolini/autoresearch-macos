# Benchmark: Misleading Comments

## Input Code

```python
def calculate_average(numbers):
    # Sort the list in descending order
    total = 0
    # Multiply each number by 2
    for n in numbers:
        total += n
    # Return the maximum value
    return total / len(numbers) if numbers else 0
```

## Expected Behaviors

The explanation MUST:
1. Identify this as Python
2. Correctly explain it calculates the average (sum / count)
3. **CRITICAL**: Ignore or call out the misleading comments
4. NOT claim it sorts, multiplies by 2, or returns maximum
5. Note the edge case handling for empty list

## Scoring

- 1.0: Correctly explains the code, ignores/flags misleading comments
- 0.5: Mostly correct but influenced by one misleading comment
- 0.0: Explanation follows the comments instead of the code
