# OMX Task: Train.py-Only Local Quality Mode

You are in train.py-only local quality mode for frozen Variant 1.

Rules:

- edit train.py only
- do not edit env.py
- do not edit eval.py
- do not edit scripts/local_runner.py
- do not change benchmark semantics
- use smoke run only for breakage checks
- use 800-episode runs as the scientific gate
- optimize ToMCoordScore without worsening deadlock
- preserve belief-guided switching under ambiguity

Primary comparison target:

- current incumbent snapshot

Preferred directions:

- belief-to-policy gating
- deadlock-aware micro-improvements
- lightweight auxiliary losses
- observation-history tuning

Issue-note habit:

- whenever a weakness or hypothesis is identified, record it in two forms:
  - a short human-readable label
  - a more detailed mechanism description
- example short labels:
  - `softness is ambiguous`
  - `cooperative moderate cases over-probe`
  - `false_friend is a recurring weakness`
- keep using the short label alongside the detailed description in summaries and follow-up notes so humans can reason about the failure family quickly

Forbidden directions:

- benchmark redesign
- multi-file refactors
- Azure work
- dashboard work
