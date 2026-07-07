# ask confidence + abstain — blind eval

Date: 2026-07-05. DB: copy of the live `~/.oracle/oracle.db` (23,448 live
facts), run with `ORACLE_HOME` pointed at the copy (ask writes reinforcement
+ traces; the live DB is never touched).

## Signal

Computed post-hoc from the investigation `ask` already ran — no extra LLM
call (`confidence.go`):

| feature | definition |
|---|---|
| cov | min(#distinct valid cited fact ids / 3, 1) |
| val | valid cited ids / total `[N]` citation tokens (0 if none) |
| tier | mean over cited of `0.5*fact.confidence + 0.5*evTier` (verified 1.0, reported 0.8, else 0.6) |
| agr | fraction of cited facts with NO live `contradicts` edge |
| ret | mean over the reasoner's searches of `min(topHitRRFScore/0.03, 1)`; a search returning nothing contributes 0 |

**base = 0.30·cov + 0.25·val + 0.20·tier + 0.15·agr + 0.10·ret**
**confidence = base × 0.4 if the answer LEADS (first ~160 chars) with
absence/unknown language, else base** (`leadsWithDoubt`).

The damp is the load-bearing part: grounded negatives ("we don't use
Elasticsearch; search is X [ids]") cite real facts and score ~0.9 on the
base formula while the reasoner itself reports no coverage. Lead-only, so
mid-answer caveats on confident answers don't damp (verified: a05/a08/a21
carry caveat words past the lead and are untouched).

If confidence < **0.45** the answer is prefixed
`LOW CONFIDENCE — likely not covered:`. The reasoner's own text states
what's missing; on tool-budget exhaustion `ask` now forces a final no-tools
answer with a system nudge ("say exactly what the graph does not contain")
instead of erroring — found via probe u01, which previously burned all 8
rounds and returned a hard error. CLI prints `confidence: 0.NN`; `GET /ask`
returns `confidence`, `abstained`, `features`.

## Probes

40 questions in `eval/ask_confidence_probes.json`:

- **25 answerable**, mined by hand from live facts in the DB copy (probe file not committed — it contains real session content; mine your own); each has
  `expect` substrings from the gold fact (any case-insensitive match =
  correct).
- **15 unanswerable**, about topics verified ABSENT before writing the
  question (FTS count = 0 live matches: elasticsearch, react native,
  snowflake, rabbitmq, unity, flutter, heroku, laravel, solana, ethereum,
  airflow, hadoop, vue, okta, neo4j).

Fixed split: **build** = a01–a13 + u01–u07 (20), **holdout** = a14–a25 +
u08–u15 (20). Features + damp markers + threshold were frozen on the build
split before any holdout unanswerable was inspected; holdout was scored
once. (Two generic markers — "not have", "don't find" — were added after
freezing, motivated by fresh smoke-test answers, not holdout probes;
verified they change no holdout row's outcome.)

Scoring: answerable = expect-match; unanswerable correct = abstained OR a
grounded negative (whole answer explicitly denies coverage/use).
**Confidently wrong** = answered above threshold with neither.

## Results

Threshold sweep on build was flat 0.40–0.50 at 13/13 answered, 7/7
abstained → threshold 0.45.

| metric | build | holdout |
|---|---|---|
| answerable answered | 13/13 | 11/12 (a23 abstained; its answer was still correct) |
| answered-and-correct | 13/13 | 11/11 |
| unanswerable abstained | 7/7 | 6/8 (u08, u14 answered as grounded negatives) |
| **confidently wrong** | **0** | **0** |

Calibration (confidence bucket → behavior-correct / n):

| bucket | build | holdout |
|---|---|---|
| 0.0–0.2 | – | 1/1 |
| 0.2–0.4 | 7/7 | 6/6 |
| 0.6–0.8 | 6/6 | 6/6 |
| 0.8–1.0 | 7/7 | 7/7 |

No row lands in 0.4–0.6: the signal is bimodal (cited+confident vs
damped/uncited), which is what makes the threshold insensitive.

Known limits: the damp is a phrase list — a grounded negative phrased in a
novel way rides high (u08/u14 did; both were still correct negatives, so
not confidently-wrong). The base formula measures citation quality, not
answer-question fit; a fluent hallucination citing real-but-irrelevant
facts would score high. None was observed in 40 probes + 3 smoke runs.

## Repro

```
mkdir -p /tmp/ohome && cp ~/.oracle/oracle.db ~/.oracle/config /tmp/ohome/
cd ~/code/oracle && go build -o /tmp/oracle-bin .
ORACLE_HOME=/tmp/ohome /tmp/oracle-bin askeval -out /tmp/askeval_rows.jsonl
```

~40 ask calls (frontier-LLM tool loop), ≈45 min, a few dollars. `askeval` prints
per-split summaries, the calibration table, and the build threshold sweep.
