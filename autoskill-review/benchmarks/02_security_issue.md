# Benchmark: Security Issue

## Input Diff

```diff
diff --git a/src/api/files.js b/src/api/files.js
+app.get('/download', (req, res) => {
+  const filename = req.query.file;
+  const filepath = path.join('/var/uploads', filename);
+  res.sendFile(filepath);
+});
```

## Expected Behaviors

The review MUST:
1. **CRITICAL**: Identify path traversal vulnerability
2. Mark it as blocker/critical severity
3. Explain the attack: `?file=../../etc/passwd`
4. Suggest fix: validate filename, use path.basename, or allowlist
5. NOT approve or treat as minor

## Scoring

- 1.0: Catches path traversal with blocker severity and fix
- 0.5: Mentions security concern but misses path traversal specifically
- 0.0: Approves or misses the security issue
