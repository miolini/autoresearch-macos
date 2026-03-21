# Benchmark: React Hook

## Input Code

```jsx
function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}
```

## Expected Behaviors

The explanation MUST:
1. Identify this as React/JSX
2. Identify it as a custom hook
3. Explain debouncing concept (delays updates)
4. Note the cleanup function (clearTimeout)
5. Mention the dependency array [value, delay]

## Scoring

- 1.0: All 5 behaviors present
- 0.5: 3-4 behaviors present
- 0.0: 2 or fewer behaviors present
