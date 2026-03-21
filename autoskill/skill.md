---
name: code-explainer
description: Explains what a piece of code does in plain language
version: 0.1.0
---

# Code Explainer

Takes code as input and produces a clear, accurate explanation.

## Instructions

When given code to explain:

1. **Identify the language** and any frameworks/libraries used
2. **State the purpose** in one sentence
3. **Walk through the logic** step by step, noting:
   - Control flow (conditions, loops, recursion)
   - Side effects (API calls, state changes, cleanup/teardown)
   - Dependencies and parameters that affect behavior
4. **Note any edge cases** or potential issues (including security concerns)
5. **Summarize** what a caller/user would experience

## Output Format

```
## Language
[language name]

## Purpose
[one sentence summary]

## How It Works
[step-by-step explanation]

## Edge Cases
[potential issues or gotchas]

## Summary
[what the user experiences]
```

## Constraints

- Be accurate — don't guess at behavior you're unsure of
- Be concise — no unnecessary words
- Use plain language — avoid jargon unless explaining it
