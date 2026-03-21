# Benchmark: Obfuscated FizzBuzz

## Input Code

```python
p=print;r=range
for i in r(1,101):p(i%3//2*'Fizz'+i%5//4*'Buzz'or i)
```

## Expected Behaviors

The explanation MUST:
1. Identify this as Python
2. **CRITICAL**: Recognize this as FizzBuzz despite obfuscation
3. Explain the modulo logic (divisible by 3 → Fizz, by 5 → Buzz)
4. Note the integer division trick (i%3//2 gives 1 when i%3==0, else 0)
5. Explain the `or i` fallback for numbers not divisible by 3 or 5

## Scoring

- 1.0: Correctly identifies FizzBuzz and explains the tricks
- 0.5: Identifies FizzBuzz but doesn't fully explain the obfuscation
- 0.0: Fails to recognize it as FizzBuzz
