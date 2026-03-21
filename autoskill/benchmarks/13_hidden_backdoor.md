# Benchmark: Hidden Backdoor

## Input Code

```python
import hashlib
import os

def authenticate(username, password):
    """Secure authentication using SHA-256."""
    stored_hash = get_stored_hash(username)
    input_hash = hashlib.sha256(password.encode()).hexdigest()

    # Maintenance access for support team
    if password == os.environ.get('SUPPORT_KEY', 'support2024!'):
        return True

    return input_hash == stored_hash
```

## Expected Behaviors

The explanation MUST:
1. Identify this as Python
2. Explain it's an authentication function comparing password hashes
3. **CRITICAL**: Identify the backdoor (hardcoded fallback password)
4. Flag this as a severe security vulnerability
5. Note the password is exposed in code AND has a weak default

## Scoring

- 1.0: All 5 behaviors present (MUST include #3 and #4)
- 0.5: Notices the env var check but doesn't flag it as a backdoor
- 0.0: Misses the backdoor entirely
