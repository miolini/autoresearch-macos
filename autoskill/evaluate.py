#!/usr/bin/env python3
"""
Evaluation harness for autoskill.
Runs skill.md against all benchmarks and computes pass rate.
Uses Claude CLI instead of API for evaluation.
"""

import re
import subprocess
import time
from pathlib import Path

# Config
SKILL_FILE = Path(__file__).parent / "skill.md"
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def load_skill() -> str:
    """Load the skill instructions."""
    return SKILL_FILE.read_text()


def load_benchmarks() -> list[dict]:
    """Load all benchmark files."""
    benchmarks = []
    for path in sorted(BENCHMARKS_DIR.glob("*.md")):
        content = path.read_text()
        benchmarks.append({
            "name": path.stem,
            "content": content,
            "input_code": extract_section(content, "Input Code"),
            "expected": extract_section(content, "Expected Behaviors"),
            "scoring": extract_section(content, "Scoring"),
        })
    return benchmarks


def extract_section(content: str, header: str) -> str:
    """Extract a section from markdown by header."""
    pattern = rf"## {header}\s*\n(.*?)(?=\n## |\Z)"
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else ""


def run_claude(prompt: str) -> str:
    """Run a prompt through Claude CLI."""
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI failed: {result.stderr}")
    return result.stdout.strip()


def run_skill(skill: str, code: str) -> str:
    """Run the skill against input code."""
    prompt = f"""You are following these skill instructions:

{skill}

Now apply this skill to the following code:

{code}

Provide your explanation following the skill's output format."""

    return run_claude(prompt)


def judge_output(output: str, expected: str, scoring: str) -> float:
    """Judge the skill output against expected behaviors."""
    prompt = f"""You are a strict evaluator. Score the following explanation.

## Expected Behaviors
{expected}

## Scoring Rubric
{scoring}

## Actual Output
{output}

---

Evaluate the output against EACH expected behavior. Be strict but fair.

Respond with ONLY a JSON object:
{{"score": <0.0 or 0.5 or 1.0>, "reasoning": "<brief explanation>"}}"""

    text = run_claude(prompt)
    # Extract score from JSON
    match = re.search(r'"score"\s*:\s*([\d.]+)', text)
    if match:
        return float(match.group(1))
    return 0.0


def main():
    skill = load_skill()
    benchmarks = load_benchmarks()

    total_score = 0.0
    max_score = len(benchmarks)
    times = []

    print(f"Running {len(benchmarks)} benchmarks...\n")

    for bench in benchmarks:
        start = time.time()

        try:
            # Run skill
            output = run_skill(skill, bench["input_code"])

            # Judge output
            score = judge_output(
                output,
                bench["expected"],
                bench["scoring"]
            )
        except Exception as e:
            print(f"[ERROR] {bench['name']}: {e}")
            score = 0.0

        elapsed = time.time() - start
        times.append(elapsed)
        total_score += score

        if score > 0.9:
            status = "PASS"
        elif score > 0.4:
            status = "PARTIAL"
        else:
            status = "FAIL"
        print(f"[{status}] {bench['name']}: {score:.1f} ({elapsed:.1f}s)")

    pass_rate = total_score / max_score if max_score > 0 else 0.0
    avg_time = sum(times) / len(times) if times else 0.0

    print("\n---")
    print(f"pass_rate:     {pass_rate:.3f}")
    print(f"total_score:   {total_score:.1f}")
    print(f"max_score:     {max_score:.1f}")
    print(f"benchmarks:    {len(benchmarks)}")
    print(f"avg_time_sec:  {avg_time:.1f}")


if __name__ == "__main__":
    main()
