# Benchmark: Performance Issue

## Input Diff

```diff
diff --git a/src/api/users.js b/src/api/users.js
+async function getUsersWithPosts(userIds) {
+  const results = [];
+  for (const id of userIds) {
+    const user = await db.users.findById(id);
+    const posts = await db.posts.find({ authorId: id });
+    results.push({ ...user, posts });
+  }
+  return results;
+}
```

## Expected Behaviors

The review MUST:
1. Identify the N+1 query problem
2. Mark as warning (not blocker, but significant)
3. Explain: sequential queries in loop = O(n) database calls
4. Suggest batching: findByIds, Promise.all, or joins
5. Note this matters at scale

## Scoring

- 1.0: Identifies N+1 with appropriate severity and fix
- 0.5: Mentions performance but wrong diagnosis
- 0.0: Misses the performance issue
