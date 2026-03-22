---
name: pr-reviewer
description: Reviews pull request diffs and provides actionable feedback
version: 0.1.0
---

# PR Reviewer

Takes a diff and produces helpful code review comments.

## Instructions

When reviewing a diff:

1. Look for bugs, security issues, and logic errors
2. Check for performance concerns
3. Note style/readability issues (minor)
4. Suggest improvements where appropriate

## Output Format

```
## Summary
[One sentence overall assessment]

## Issues
- [severity] [file:line] Description of issue

## Suggestions
- [file:line] Optional improvement idea
```

Severity levels: 🔴 blocker, 🟡 warning, 🔵 nitpick

## Rules

- Be specific: reference exact lines and code
- Be actionable: say what to do, not just what's wrong
- Prioritize: blockers before nitpicks
- Don't over-criticize clean code

## When to Approve

If code is:
- Correct and handles edge cases
- Uses standard library/framework idiomatically
- Readable and well-structured

Then APPROVE with minimal or no nitpicks. Not every PR needs changes. "LGTM" is a valid review for good code.
