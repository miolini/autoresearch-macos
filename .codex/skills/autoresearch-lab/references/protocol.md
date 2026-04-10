# Autoresearch Protocol

This reference captures the durable parts of the original `program.md` workflow in a form that pairs cleanly with the repo-local skill.

## Scope

- `train.py` is the experiment surface
- `prepare.py` is fixed
- No new dependencies
- The benchmark remains a fixed 5-minute training budget

## Setup Expectations

Before a new experiment lane starts:

1. Pick a run tag and use a dedicated branch such as `autoresearch/<tag>`.
2. Verify cached data and tokenizer files exist under `~/.cache/autoresearch/`.
3. Ensure `results.tsv` exists with this header:

```text
commit	val_bpb	memory_gb	status	description
```

## Evaluation Contract

- The objective metric is `val_bpb`; lower is better.
- `peak_vram_mb` is a secondary operational metric.
- Simpler wins when metric changes are negligible.
- A small improvement is not automatically worth a large complexity increase.

## Iteration Loop

For each experiment:

1. Start from the current accepted commit.
2. Edit `train.py`.
3. Commit the experiment.
4. Run `train.py` and capture output to `run.log`.
5. Parse the emitted summary block.
6. Record the result in `results.tsv`.
7. Keep the commit only when it meaningfully improves the incumbent or clearly simplifies the code without regression.

## Crash And Timeout Policy

- If a run exceeds 10 minutes, treat it as a failure.
- If the log does not contain a final summary, treat the run as a crash until proven otherwise.
- Easy mechanical issues can be fixed and rerun.
- Fundamentally bad ideas should be logged as `crash` or `discard` and abandoned.

## Logging Rules

- `results.tsv` is tab-separated
- Descriptions should not contain tabs or newlines
- Crash rows use `0.000000` for `val_bpb` and `0.0` for memory when no summary is available

## Keep/Discard Guidance

- Keep: lower `val_bpb`, or materially simpler code with equal performance
- Discard: worse `val_bpb`, negligible gains with ugly complexity, or unstable ideas
- Crash: run failed, timed out, or never produced the summary footer

