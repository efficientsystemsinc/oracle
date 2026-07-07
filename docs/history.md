# History — how the graph got trustworthy

A condensed log of the quality work that shaped oracle's current design.
(Detailed run notes lived here as separate docs; this replaces them.)

## The false-supersession problem (2026-07)

`oracle judgeaudit` samples supersession pairs and asks an independent
skeptic judge whether replacing OLD with NEW was actually correct. The first
audit of 100 random historical pairs found an upper-bound **75% false
supersession rate**. Dominant failure modes:

- **Different aspect, same entity** — NEW covers a related but distinct
  aspect of the same system, so superseding silently drops still-true detail
  (deployment coordinates replaced by an unrelated failure note; a specific
  operational rule swallowed by a general one).
- **Date inversion** — an older fact retiring a newer one, because the judge
  never saw `valid_from`.

Fixes: the judge prompt now carries `valid_from` + evidence tier for both
facts, forbids date-inverted supersession, and requires true replacement
(same topic, newer state) rather than mere relatedness.

## The truth wave (2026-07-05)

One end-to-end repair pass over the whole graph:

1. **Judge upgrade** (dates + evidence tiers in context).
2. **`oracle repair`** — re-audited all ~15k historical supersession pairs
   with the upgraded judge (24 workers). Verdicts: ~58% of historical
   supersessions were wrong; ~9.4k facts were reopened. Live fact count grew
   ~22.6k → ~32.6k.
3. **`oracle sweep`** — recurring consolidation: per (entity, kind),
   semantically-near live facts far apart in world time go to the judge
   queue, catching supersessions ingest missed.
4. **`oracle referee`** — resolved live contradiction pairs into
   supersede / different-scope / unresolved; `oracle conflicts` lists the
   remainder for human burn-down.
5. **Paraphrase index** — LLM paraphrases of the top-mass live facts,
   embedded and FTS-indexed, to close the vocabulary gap in retrieval.
6. **Fixture freeze + probe suites** — a frozen fixture DB plus 1000
   generated probes (chain / paraphrase / time-current / time-as-of) turned
   memory quality into a number that gates changes. Post-wave: original
   probes 15/15, mined-150 132/150, suite-1k 767/1000.

Known gaps at the end of the wave, in priority order: as-of ranking under
reopened near-duplicate crowds (weakest probe category), a few hundred
unresolved conflicts, and distilling the ~16k repair verdicts into the small
local cross-encoder judge (see `truth_design.md`).

## Design research

`truth_design.md` records the 2026-07 deep-research sweep that validated the
bi-temporal + ingest-time-judge architecture against production memory
systems and the literature, and warned off heavyweight truth-discovery
machinery. Its build list (repair / sweep / referee / temporal typing /
eval-gated CI) is what the truth wave shipped.
