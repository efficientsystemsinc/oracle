---
name: oracle
description: Query the local oracle fact graph — bi-temporal memory extracted from every Claude Code/codex session on this machine. Use when starting work in a repo (pull the standing brief), when the user references past work, decisions, infrastructure, or "that thing we did", before assuming how a box/service/deployment is configured, or for history questions ("what changed", "as of June", "why did we choose X"). Needs the oracle daemon on 127.0.0.1:4141 or the oracle CLI on PATH.
---

# oracle — cross-session memory graph

oracle watches all agent session logs on this machine, extracts durable facts
with an LLM, and serves them via hybrid (lexical+semantic) search with
time-decay. Facts carry a kind (decision | fact | gotcha | preference | status
| todo), age, confidence, and repo. Nothing is deleted: replaced facts are
superseded and remain queryable by date.

## The main verb: `ask`

`oracle ask` is the primary interface — a multi-hop reasoner that plans
retrieval FOR you: it reformulates searches, follows entities across the
graph, compares time periods, then answers with [fact-id] citations and a
calibrated confidence (it abstains rather than guesses).

```sh
oracle ask "why did we move off the old database, and what broke?"
oracle ask --json "..."        # {answer, confidence, abstained, sources[]} — you get the answer AND the raw facts
oracle ask --as-of 2026-06-01 "..."   # answered from what was true then
```

As an agent you are tempted to run `query` and reason over raw facts yourself.
That works for one-hop lookups — but `ask` does the multi-hop retrieval you
would otherwise have to orchestrate across several calls (entity hops, query
reformulation, historical comparison). Rule of thumb:

- **`ask`** when the question is why/how/what-happened, spans topics or time,
  or a first `query` came back thin. Use `--json` and read `sources` — you get
  better retrieval than your own single query AND the facts to reason over.
- **`query`** for quick single-hop lookups where you know the noun ("cato ssh
  user", "pgbouncer setting") — it answers in ~10ms.
- **`brief --repo X`** once, when entering a repo.

## Quick reference

```sh
oracle query "asyncpg pgbouncer setting"      # ranked facts, ~10ms (-k N, --repo X, --as-of DATE)
oracle brief --repo perseus                    # standing brief for a repo, by kind
oracle graph entity kmlv                       # everything known about one named thing
oracle graph relations perseus --hops 2        # typed relations + co-mentions
oracle query "asyncpg pgbouncer setting"     # ranked facts (-k N, --repo X, --as-of DATE)
oracle ask "why did we abandon vllm?"        # multi-hop reasoner; cites [fact-id]
oracle brief --repo quasar                  # standing brief for a repo, by kind
oracle entity atlas01                           # everything known about one named thing
oracle graph quasar --hops 2                # typed relations + co-mentions
oracle metric                                # numeric time series (no arg = list)
oracle narrative <entity-or-repo>            # chronological story of a subject
```

HTTP equivalents on `http://127.0.0.1:4141`:
`GET /query?q= /ask?q= /brief?repo= /entity/{name} /graph/{name} /metrics/{name} /narrative/{name} /health`.

Prefer `query` for a quick lookup (fast, no LLM); use `ask` when the question
needs multiple hops, comparison across time, or metric series (slower, LLM).

## Reading results

- `[kind] (repo, age-days, mass) statement` — prefer newer, heavier facts.
- `status` and `todo` facts decay in days; treat old ones as probably stale.
- Conflicting facts: prefer the lower `age_days`; inspect history via `--as-of`.
- `ask` cites fact ids like `[123]`; keep the ids when relaying precise claims.

## Boundaries and failure

- Not a code-search tool — use grep/Glob for code. oracle is for knowledge
  that lives across sessions, not in files.
- Facts are extracted claims, not gospel: for destructive actions, verify
  load-bearing facts (an IP, a "safe to delete") against the live system first.
- If it's down: `systemctl --user status oracle`, or run `oracle up`
  (db at `~/.oracle/oracle.db`).
