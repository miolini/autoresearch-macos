# Provenance Forensic Timeline (2026-04-09)

This file exists so the lineage of the current ToM runs does not have to be
reconstructed again from memory.

Canonical project root:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team`

## Executive Summary

The current `modal/tom-140k-modal-results-v2` branch is a **V2 continuation
branch warm-started from the older promoted `auxhead-lite` `800`-episode
incumbents**, not from the later exact local V2 `800` selected-model
checkpoints.

The first exact archive of the local V2 `800` warm starts was created on
2026-04-09 under:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800`

## Terms

- `family`: a model lineage such as `auxhead-lite`
- `checkpoint`: one exact saved `.pt` model state
- `archive`: a curated on-disk location used as a warm-start source
- `continuation regime`: the training/evaluation setup used after warm start

## Timeline

### 2026-04-05 20:58:51 to 20:59:12

Legacy deadlock-micro incumbents existed and were documented under:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/seed7/selected_model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/seed11/selected_model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/INCUMBENT_NOTE.md`

These were the earlier incumbent checkpoints before auxhead-lite promotion.

### 2026-04-05 22:24:22 to 22:24:44

Auxhead-lite incumbents for `seed7` and `seed11` were promoted into the archive:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed7/selected_model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed11/selected_model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/INCUMBENT_NOTE.md`

This created the warm-start archive later used by the Modal continuation path.

### 2026-04-07 18:31:37 to 18:31:41

The older `130k` long-run branch completed:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-130k-modal-results/seed7/target-130000/run_summary.json`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-130k-modal-results/seed11/target-130000/run_summary.json`

These run summaries record the warm start as:

- `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
- `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

So the old `130k` branch definitely used the promoted `auxhead-lite` archive.

### 2026-04-07 21:17:11 to 22:07:38

The older `140k` long-run branch completed:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-140k-modal-results/seed7/target-140000/run_summary.json`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-140k-modal-results/seed11/target-140000/run_summary.json`

These run summaries also record the warm start as:

- `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
- `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

So the old `140k` branch also used the promoted `auxhead-lite` archive.

### 2026-04-08 02:47:04 to 02:48:47

New local V2 `800` selected-model checkpoints were produced:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed7/selected_model/model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed11/selected_model/model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed7/candidate_metrics/metrics.json`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed11/candidate_metrics/metrics.json`

These were later local runs than the promoted `auxhead-lite` archive and have
different checkpoint hashes from the promoted archive entries.

### 2026-04-08 to 2026-04-09 (the key divergence)

The newer local V2 `800` checkpoints were **not promoted into the live Modal
warm-start archive** before the V2 `140k` Modal continuation branch ran.

The runner still pointed at the promoted `auxhead-lite` archive under
`modal/tom-experiment-incumbent/auxhead-lite`.

### 2026-04-09 01:44:47 to 01:44:54

The V2 `140k` branch completed:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-140k-modal-results-v2/seed7/target-140000/run_summary.json`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-140k-modal-results-v2/seed11/target-140000/run_summary.json`

These run summaries record the warm start as:

- `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
- `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

Therefore the V2 `140k` branch is:

- `family`: `auxhead-lite`
- `continuation regime`: V2
- `warm-start archive`: legacy promoted `auxhead-lite`
- **not** an exact continuation of the local V2 `800` selected-model checkpoints

### 2026-04-09 18:13:44

Local auxhead-lite checkpoints for `seed13` and `seed17` were staged into the
legacy `auxhead-lite` archive:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed13/selected_model.pt`
- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed17/selected_model.pt`

Their hashes match the corresponding local source files from:

- `logs/local-run-v1-auxhead-lite-s13/selected_model/model.pt`
- `logs/local-run-v1-auxhead-lite-s17/selected_model/model.pt`

### 2026-04-09 18:48:18 to 18:48:46

Two corrective changes were made:

1. The Modal runner was patched so it can:
   - discover packaged seeds automatically
   - record incumbent family
   - record checkpoint hashes
   - use configurable archive/output families

   File:
   - `/Users/stephenbeale/Projects/ToM_AI_Research_Team/scripts/modal_auxhead_lite_runner.py`

2. A new exact archive was created from the local V2 `800` selected models:

   - `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800`
   - `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800/PROVENANCE_MANIFEST.json`

This archive is the first clean warm-start source that exactly matches the local
V2 `800` selected-model checkpoints for `seed7` and `seed11`.

## Checkpoint Hashes

### Legacy deadlock-micro incumbents

- `seed7`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/seed7/selected_model.pt`
  - timestamp: `2026-04-05 20:58:51`
  - SHA1: `316dc5b2ae7b7439aa782932381f23cf87ab7d0a`

- `seed11`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/seed11/selected_model.pt`
  - timestamp: `2026-04-05 20:58:52`
  - SHA1: `6bd2f090cc61190d2e57d9e08888fd382ec17bea`

### Legacy promoted auxhead-lite incumbents

- `seed7`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed7/selected_model.pt`
  - timestamp: `2026-04-05 22:24:22`
  - SHA1: `8663e34f80ad8dafeba560e8a6ee0bcdf33c63d7`

- `seed11`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed11/selected_model.pt`
  - timestamp: `2026-04-05 22:24:22`
  - SHA1: `b9bba4dac2cdbcfebc0d708fe3c74617a6bd2de7`

### Later local V2 `800` selected-model checkpoints

- `seed7`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed7/selected_model/model.pt`
  - timestamp: `2026-04-08 02:47:04`
  - SHA1: `9adca7d63617a6bd7cdb8b07a020ec64025a3faf`

- `seed11`
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v2-modal-seed11/selected_model/model.pt`
  - timestamp: `2026-04-08 02:48:46`
  - SHA1: `b8566ba23a2bc077e376f9d07dcb154b96190a15`

### Local auxhead-lite `seed13` / `seed17` selected-model checkpoints

- `seed13`
  - source path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v1-auxhead-lite-s13/selected_model/model.pt`
  - source timestamp: `2026-04-05 22:25:40`
  - source SHA1: `2403c2f31ccd731a7ba0acd58717d7ac2525a329`
  - archived path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed13/selected_model.pt`
  - archived timestamp: `2026-04-09 18:13:44`
  - archived SHA1: `2403c2f31ccd731a7ba0acd58717d7ac2525a329`

- `seed17`
  - source path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/logs/local-run-v1-auxhead-lite-s17/selected_model/model.pt`
  - source timestamp: `2026-04-05 22:25:40`
  - source SHA1: `61dc9c2538393a0d9f57219dc02ae199160699fe`
  - archived path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite/seed17/selected_model.pt`
  - archived timestamp: `2026-04-09 18:13:44`
  - archived SHA1: `61dc9c2538393a0d9f57219dc02ae199160699fe`

### Clean exact V2-local800 archive

- `seed7`
  - archived path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800/seed7/selected_model.pt`
  - archived timestamp: `2026-04-09 18:48:38`
  - archived SHA1: `9adca7d63617a6bd7cdb8b07a020ec64025a3faf`

- `seed11`
  - archived path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800/seed11/selected_model.pt`
  - archived timestamp: `2026-04-09 18:48:38`
  - archived SHA1: `b8566ba23a2bc077e376f9d07dcb154b96190a15`

- manifest
  - path: `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800/PROVENANCE_MANIFEST.json`
  - timestamp: `2026-04-09 18:48:46`
  - SHA1: `c0d3802b805081542ea982dde31d2d64aa1f0af4`

## Branch Results and Warm-Start Sources

### Old `130k` branch

- warm-start source recorded in run summaries:
  - `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
  - `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

- results:
  - `seed7`: `ToMCoordScore 0.23559881878531513`
  - `seed11`: `ToMCoordScore 0.22581225267628627`

### Old `140k` branch

- warm-start source recorded in run summaries:
  - `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
  - `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

- results:
  - `seed7`: `ToMCoordScore 0.27929565785537974`
  - `seed11`: `ToMCoordScore 0.20141472197593688`

### Current V2 `140k` branch

- warm-start source recorded in run summaries:
  - `/root/incumbent/auxhead-lite/seed7/selected_model.pt`
  - `/root/incumbent/auxhead-lite/seed11/selected_model.pt`

- results:
  - `seed7`: `ToMCoordScore 0.2959216451784249`
  - `seed11`: `ToMCoordScore 0.38870678285503113`

## What This Means

1. The current V2 `140k` branch is a **real and valid branch**.
2. It was **not** warm-started from the later exact local V2 `800` selected-model
   checkpoints.
3. It **was** warm-started from the older promoted `auxhead-lite` archive and
   then continued under the V2 training/evaluation regime.
4. The new archive `auxhead-lite-v2-local800` is the first archive that exactly
   matches the local V2 `800` selected-model checkpoints for `seed7` and
   `seed11`.

## What To Trust Going Forward

If an exact local V2 `800` warm-start lineage is needed, use only:

- `/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/auxhead-lite-v2-local800`

and record:

- incumbent family
- checkpoint hash
- output family

in the run summary.

## 2026-04-10 Replication Addendum: `check2` / `check3`

Two later duplicate runs were executed from the corrected exact V2 local `800`
archive using the dedicated duplicate runner:

- [`check2-140k-20260410`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check2-140k-20260410)
- [`check3-140k-20260410`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/modal/tom-experiment-incumbent/check3-140k-20260410)
- runner:
  [`scripts/modal_v2b_runner.py`](/Users/stephenbeale/Projects/ToM_AI_Research_Team/scripts/modal_v2b_runner.py)

These later reruns are important because they change the answer to the original
provenance question.

### What They Resolve

The original issue was:

- the earlier V2 `140k` branch was a real branch, but not an exact continuation
  of the intended local V2 `800` checkpoints

`check2` and `check3` materially improve that situation because their run
summaries record warm starts from:

- `/root/incumbent/auxhead-lite-v2-local800/seed7/selected_model.pt`
- `/root/incumbent/auxhead-lite-v2-local800/seed11/selected_model.pt`

So the “wrong warm-start archive” provenance problem is no longer the main
concern for these duplicate runs.

### What They Show

At the branch level:

- `check2` is the stronger duplicate branch
- `check3` is weaker, especially on `seed11`
- but both runs remain well above baseline and preserve the same qualitative
  family pattern, including:
  - `late_disambiguation` as a strength family
  - `assert_or_yield` as the main unresolved weak family

This means:

- reproducible configuration: yes
- exact deterministic replay: no

### How To Interpret Them

For provenance purposes, `check2` / `check3` are enough to support the claim
that the current duplicate-run configuration is real, repeatable, and now tied
to the intended exact V2 local `800` archive.

They are not enough to support the stronger claim that the branch is now
low-variance or numerically deterministic.

### Practical Trust Update

Going forward, trust `check2` / `check3` for:

- exact archive lineage
- corrected warm-start source
- repeated behavioral regime

Do not over-claim them as:

- proof of exact replay
- proof that one run fully characterizes the branch
