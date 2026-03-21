# Benchmark: Async Fetch

## Input Code

```javascript
async function fetchUser(id) {
  try {
    const response = await fetch(`/api/users/${id}`);
    if (!response.ok) throw new Error('Not found');
    return await response.json();
  } catch (error) {
    console.error('Failed to fetch user:', error);
    return null;
  }
}
```

## Expected Behaviors

The explanation MUST:
1. Identify this as JavaScript
2. Mention async/await pattern
3. Explain it fetches user data from an API
4. Note the error handling (try/catch)
5. Mention it returns null on failure

## Scoring

- 1.0: All 5 behaviors present
- 0.5: 3-4 behaviors present
- 0.0: 2 or fewer behaviors present
