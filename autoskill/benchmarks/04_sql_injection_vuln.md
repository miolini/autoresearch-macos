# Benchmark: SQL Injection Vulnerability

## Input Code

```python
def get_user(username):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    return db.execute(query)
```

## Expected Behaviors

The explanation MUST:
1. Identify this as Python
2. Explain it queries a database for a user
3. **CRITICAL**: Identify the SQL injection vulnerability
4. Mention f-string/string formatting is dangerous here
5. Suggest parameterized queries as the fix

## Scoring

- 1.0: All 5 behaviors present (MUST include #3)
- 0.5: 3-4 behaviors present AND includes #3
- 0.0: Missing the SQL injection warning (#3)
