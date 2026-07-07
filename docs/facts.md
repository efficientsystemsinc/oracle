# The fact model

Everything in oracle reduces to one table: `facts`. A fact is an atomic,
standalone statement extracted from a session — readable with zero context,
naming its systems and numbers explicitly. Each carries a `kind`
(decision | fact | gotcha | preference | status | todo), extracted entities,
central files, and the extractor's confidence.

## Two clocks (bitemporality)

Every fact lives on two independent timelines:

- **World time — `valid_from`.** When the thing *became true* out in the
  world. Taken from the session event's timestamp, not from when oracle read
  it. A backfilled fact about an October decision gets an October
  `valid_from`, even if ingested months later.
- **Transaction time — `recorded_at`, `superseded_at`.** When *oracle* learned
  the fact, and when oracle learned it had been replaced. Pure bookkeeping of
  the database's own knowledge.

The split matters because ingestion lag is arbitrary. "What was true on June
20?" must be answered by world time; `recorded_at` would say "nothing" for
anything backfilled after June 20. So `--as-of T` selects:

```
valid_from <= T
AND (superseded_by IS NULL
     OR superseder.valid_from > T)
```

i.e. the fact had already become true at T, and nothing that replaced it had
become true yet. This reconstructs the state of knowledge at any past moment —
ask about AST language support as-of June 20 and you get the "12 languages,
asymmetric" era; ask today and you get the 26-language parity facts.

## Updating: supersession, never deletion

Facts are append-only. Nothing is ever deleted or edited. The only mutation a
fact can suffer is being **superseded**:

1. Each new fact is compared against its nearest live neighbours
   (FTS match, same repo, same kind, top 3).
2. An LLM judge decides: does the new fact *replace* an old one — same topic,
   newer state, or an outright duplicate? Different aspects of the same system
   do not supersede.
3. If yes, the old fact gets `superseded_at = now`, `superseded_by = new id`,
   and a `supersedes` edge. It drops out of default search but remains fully
   queryable through `--as-of` and the supersession chain on `/facts/{id}`.

Over the full backfill this consolidated 39% of all extracted facts — the
graph converges on current truth by *closing validity intervals*, not by
forgetting that the old truth ever held.

## Decay: forgetting as attention, not erasure

Superseded facts are gone from defaults; live facts merely *fade*. Each fact
has a retrieval `mass` whose effective value halves per kind-specific
half-life, computed at read time:

```
mass_eff = 0.05 + mass · 2^(−age / half_life)

status 7d · todo 14d · fact 60d · decision 120d · gotcha 120d · preference 365d
```

The half-lives encode how each kind rots: a status line is stale in a week; a
gotcha or architectural decision stays load-bearing for months; a user
preference is near-permanent. The `0.05` floor means nothing ever reaches
zero — an old fact is always reachable if the query matches it strongly
enough; it just needs a better match to outrank fresher material.

Retrieval pushes back against decay: each time a fact is returned in the top
results its stored `mass` grows (+0.15). Useful memories are rehearsed;
untouched ones sink toward the floor. Forgetting here is a ranking prior, not
data loss.

## The invariant

At any moment, for any topic, the graph holds exactly one *live* chain-head
fact plus its full superseded history, each stamped with when it was true and
when we knew it. Ask now → the head, weighted by decay. Ask as-of T → whichever
link in the chain was true at T. Nothing else in oracle is more important than
preserving that invariant.
