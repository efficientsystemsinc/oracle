# Keeping oracle true: research-backed design (2026-07-05)

Deep-research sweep (102 agents, 3-vote adversarial verification) over 2023-26
production memory systems + literature. Verdicts, and what oracle does about each.

## What the research validated about our current design

1. **Bi-temporal + ingest-time LLM supersession is THE shipped pattern.**
   Zep/Graphiti (the production reference) uses four timestamps per edge
   (transaction created/expired + event valid/invalid) and an ingest-time LLM
   that invalidates contradicted edges — structurally identical to our
   valid_from / recorded_at / superseded_at chains + judge. Zep beats
   full-context on LongMemEval (63.8 vs 55.4 gpt-4o-mini) at ~90% lower
   latency. [arxiv 2501.13956] → keep the architecture; it is state of the art.

2. **Embedding similarity CANNOT detect supersession** — AUROC ~0.59 (chance)
   at telling contradiction from duplication; plain RAG serves superseded
   values 15-40% of the time. [arxiv 2606.26511, direction solid / numbers
   provisional] → decay-weighted retrieval alone can never fix
   stale-served-as-current; write-path judging + structural repair are
   mandatory. Our consolidation-sweep plan is required, not optional.

3. **Per-kind half-life decay > binary current/stale** — HALO improved all
   five TKG baselines it touched. [arxiv 2505.07509] → our kind-specific
   half-lives are the right shape; refinement is per-FACT half-lives later.

## What the research warns us OFF

4. **Do not build heavyweight truth-discovery machinery.** A 12-algorithm
   benchmark: nothing clearly beats majority voting (+0.032 ± 0.070 precision
   for the best), and ALL degrade to near-random in the sparse-conflict /
   unreliable-source regime — which is exactly agent memory. [arxiv 1409.6428]
   → skip TruthFinder/Bayesian source-trust loops. Our lightweight stack
   (evidence tiers + cross-session corroboration + contradiction penalty) is
   the correct cost/benefit point. At most: asymmetric per-source trust
   nudges later.

5. **Keep-with-provenance beats force-resolve** — the Facebook KG (~500M
   statements) retains conflicting facts with provenance + confidence, drops
   only low-confidence. → the referee should prefer "both-true-different-scope"
   and "unresolved-but-visible" over forcing a winner; supersede only on
   clear temporal replacement. Never delete a side of a dispute.

## The build list this settles (in order)

- **`oracle repair`** — re-audit all supersession pairs with the upgraded
  judge (dates + evidence tiers), reopen wrongly-closed facts. Justified by
  our measured ≤75% false-supersession upper bound and by (2): the write-path
  judge is the only guard, so its historical errors must be repaired, not
  decayed away.
- **`oracle sweep`** — consolidation: per (entity, kind), semantically-near
  live facts far apart in valid_from → judge queue. Catches the supersessions
  ingest missed. Runs recurring (hygiene, not one-shot) — the "sleep-time
  reprocessing" pattern from the memory literature.
- **`oracle referee`** — resolve live contradicts pairs per (5): supersede /
  different-scope / keep-both-visible. `oracle conflicts` lists the unresolved.
- **Temporal query typing** — current-state questions hard-prefer fresh facts
  for perishable kinds; every answer stamps staleness. Stale-labeled-stale is
  fine; stale-as-current is the failure mode (2) quantifies.
- **Eval-gated CI** — frozen post-repair fixture DB; era/continuity/hit@5 gate
  every commit (LongMemEval-style memory-quality metrics as CI, which is how
  the production systems keep score). Plus weekly extraction canaries for
  model drift.

All of it is schema/pipeline work in a single Go binary + sqlite — none of it
needs infrastructure. Sources and full findings: deep-research run wf_97cff624.
