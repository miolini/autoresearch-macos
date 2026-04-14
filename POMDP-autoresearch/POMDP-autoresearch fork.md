POMDPs Autoresearch Guide

## This section references:

Chades, I., Pascal, L. V., Nicol, S., Fletcher, C. S., & Ferrer‐Mestres, J. (2021). A primer on partially observable Markov decision processes (POMDPs). Methods in Ecology and Evolution, 12(11), 2058–2072. 
DOI: 10.1111/2041-210X.13692.

## Karpathy's autoresearch is used here as a research loop template, not as a drop-in POMDP solver.

Autoresearch is a strong fit for POMDP engineering loops: solver tuning, benchmarking, code optimization, compact policy search, and interpretability tooling. The primer highlights repetitive, high-dimensional design choices under expensive evaluation, where constrained autonomous experimentation is useful.

## AR's core pattern is:

- Let an agent edit a constrained code surface.
- Run a fixed-budget experiment.
- Score the result with one comparable metric.
- Keep or discard the change.
- Repeat.

## In this repo style:

- The agent edits one file.
- It runs a fixed-time experiment.
- It compares outcomes under a stable metric.
- Instructions live in program.md.

That maps well onto the bottlenecks noted by Chades et al. (2021).

## POMDPs are hard because:

- Partial observability forces reasoning over belief states.
- Full history is too large to store explicitly.
- Belief space is continuous.
- Exact dynamic programming does not scale.

Studies show that exact, approximate, and heuristic solvers all matter, but applied users often care about interpretation and explanation in addition to raw performance.

# Application of Autoresearch for POMDPS

## 1. Optimization of POMDP-Adjacent Code

Autoresearch "single editable file + fixed evaluation budget" pattern used for:

- Belief update kernels.
- Backup operators.
- Alpha-vector pruning.
- Point sampling logic.
- Policy graph post-processing.
- Factored and MOMDP data structures.

The study frames solving POMDPs as computationally formidable. Approximate methods dominate because enumerating all reachable beliefs is infeasible. A loop that proposes code changes and benchmarks them under fixed seeds and fixed wall-clock budgets is well matched to this setting.

## 2. Tune Approximate Solvers

The study discusses grid methods, point-based methods, Perseus, Symbolic Perseus, and APPL/SARSOP. Point-based methods help because they optimize over reachable belief points instead of wasting effort on beliefs policy execution never visits.

An autoresearch loop can search over:

- Belief-point selection rules.
- Horizon and stopping criteria.
- Backup frequency.
- Pruning thresholds.
- Exploration versus exploitation in reachable-belief sampling.
- Solver-specific tolerances and factored representations.

This is especially natural for SARSOP and Perseus style workflows.

## 3. Compare Solver Settings Fairly

Autoresearch relies on a fixed time budget and one metric so experiments are directly comparable. For POMDPs, use a harness that fixes:

- Problem instance.
- Initial belief b0.
- Random seeds.
- Wall-clock budget.
- Output format.
- Evaluation rollouts.

Then compare settings by:

- Expected discounted return from simulation.
- Regret against a trusted baseline on small instances.
- Number of alpha-vectors, |Gamma|.
- Policy graph size.
- Memory footprint.
- Solve time.

This directly addresses the paper's concern that solutions should be usable and understandable, not just mathematically optimal.

## 4. Automate Benchmarking

The paper lists multiple solver families and file formats and stresses that guidance remains scarce.

An autoresearch-style agent can automate a benchmark matrix across:

- Exact versus approximate versus heuristic methods.
- Small versus large state spaces.
- Standard POMDP versus MOMDP and factored forms.
- Type 1, 2, and 3 problem classes from the paper.

A useful benchmark table should report return, runtime, alpha-vector count, and interpretability proxies.

## 5. Search for Compact Policy Representations

The study says POMDP solutions are often hard to visualize and interpret. Policy graphs can reach thousands of nodes, and alpha-vector count is a practical indicator of interpretability difficulty. It also shows some policies can be simplified into compact, human-readable rules.

An autonomous loop can optimize not only return, but also:

- Fewer alpha-vectors.
- Fewer policy graph nodes.
- Shallower decision trees extracted from policies.
- Sparse threshold rules over belief summaries.
- Stability of explanations across nearby beliefs.

This directly supports the paper's call for more interpretable solutions.

## 6. Build Interpretability Aids

Applied domains may value explanation over pure performance. Configure the agent to generate artifacts after each run:

- Policy graph simplifications.
- Dominant alpha-vector regions.
- Belief-to-action heatmaps for low-dimensional cases.
- Scenario analyses over plausible observation sequences.
- Distilled rules from policy graphs.
- Counterfactual action-switch boundaries.

Current research points to scenario analysis, factored representations, and simplified policy graphs as practical routes to understanding solutions.

## Concrete Adaptation Blueprint

- prepare.py equivalent: fixed problem definitions, simulators, rollout evaluator, and parsers for .pomdp and .pomdpx.
- train.py equivalent: the only editable file, containing solver settings, approximation logic, pruning rules, or representation-learning code.
- program.md equivalent: instructions telling the agent to optimize a weighted score such as:

```text
score = return - lambda1 * runtime - lambda2 * alpha_count - lambda3 * policy_graph_nodes
```

This preserves the core autoresearch spirit: constrained edits, repeatable scoring, and overnight iteration.

## Caveats

- Karpathy's repo is centered on small-scale LLM training, so for POMDP work you borrow methodology, not the exact software stack.
- If you optimize only for reward, interpretability can regress. Keep the objective multi-objective from day one: performance, compute, and explanation quality.
