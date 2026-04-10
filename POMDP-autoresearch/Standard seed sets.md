# Standard Seed Sets

## 3-seed quick gate

Use:

- 7
- 11
- 17

Purpose:

- Fast check for real improvement versus one-seed luck.

## 5-seed promotion gate

Use:

- 7
- 11
- 17
- 23
- 29

Purpose:

- Incumbent promotion decision.

## Optional 7-seed confidence gate

Use:

- 7
- 11
- 17
- 23
- 29
- 31
- 37

Purpose:

- High-confidence confirmation only.

---

## Episode Standard

Use:

- 800 episodes per seed.

Reason:

- Matches your current scientific gate.
- Smoke remains only for breakage, not scientific judgment.

---

## Recommended Documented Policy

### Quick-gate rule

A train.py-only change passes the quick gate if, across 3 seeds x 800 episodes:

- Mean DeadlockRate is not worse.
- Mean ToMCoordScore is higher.
- No seed has a severe deadlock regression.

Suggested guardrail:

- No seed with DeadlockRate worse than baseline by more than +0.10.

### Promotion rule

Promote to incumbent only if, across 5 seeds x 800 episodes:

- Mean DeadlockRate is not worse.
- Mean ToMCoordScore is higher.
- Mean CollisionRate is lower or equal.
- Mean SuccessRate is higher or equal.
- No catastrophic single-seed regression.

Suggested catastrophic regression rule:

- No seed with DeadlockRate delta > +0.10.
- No seed with ToMCoordScore drop < -0.05 relative to baseline.

---

## Naming Convention

For this change, use a short experiment label:

- v3omx_postevidence_reengage

Then outputs:

### Quick gate

- logs/v3omx_postevidence_reengage_seed7
- logs/v3omx_postevidence_reengage_seed11
- logs/v3omx_postevidence_reengage_seed17

### Promotion gate

- Same pattern for seeds 23 and 29.

---

## Documentation Template

### Experiment header

Experiment name: v3omx_postevidence_reengage  
Scope: train.py-only  
Benchmark: Variant 1 frozen ambiguous bottleneck  
Scientific gate: 800 episodes  
Quick gate seeds: 7, 11, 17  
Promotion seeds: 7, 11, 17, 23, 29

### Patch summary

- Tightened late_yield in non-narrow contexts so yielding is preferred only when the partner is actually pressing.
- Added soft_reengage to bias PROCEED and PROBE over WAIT and YIELD when evidence is available, margin is not narrow, partner remains soft, and belief is no longer strongly assertive.

### Per-seed results table

Seed 7:

- Decision:
- Baseline ToMCoordScore:
- Candidate ToMCoordScore:
- Baseline Deadlock:
- Candidate Deadlock:
- Baseline Collision:
- Candidate Collision:
- Baseline Success:
- Candidate Success:

Seed 11:

- Decision:
- Baseline ToMCoordScore:
- Candidate ToMCoordScore:
- Baseline Deadlock:
- Candidate Deadlock:
- Baseline Collision:
- Candidate Collision:
- Baseline Success:
- Candidate Success:

Seed 17:

- Decision:
- Baseline ToMCoordScore:
- Candidate ToMCoordScore:
- Baseline Deadlock:
- Candidate Deadlock:
- Baseline Collision:
- Candidate Collision:
- Baseline Success:
- Candidate Success:

### Aggregate summary

- Mean baseline ToMCoordScore:
- Mean candidate ToMCoordScore:
- Mean baseline DeadlockRate:
- Mean candidate DeadlockRate:
- Mean baseline CollisionRate:
- Mean candidate CollisionRate:
- Mean baseline SuccessRate:
- Mean candidate SuccessRate:

### Decision

- Quick gate: keep, discard
- Promotion: keep, discard, provisional promote

---

## Command Template

For each seed:

```bash
python scripts/local_runner.py --train-episodes 800 --seed <SEED> --output-root logs/v3omx_postevidence_reengage_seed<SEED>
```

Concrete quick-gate commands:

```bash
python scripts/local_runner.py --train-episodes 800 --seed 7 --output-root logs/v3omx_postevidence_reengage_seed7
python scripts/local_runner.py --train-episodes 800 --seed 11 --output-root logs/v3omx_postevidence_reengage_seed11
python scripts/local_runner.py --train-episodes 800 --seed 17 --output-root logs/v3omx_postevidence_reengage_seed17
```
