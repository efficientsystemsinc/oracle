# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

oracle is a bi-temporal fact graph over every AI coding-agent session on the machine. A Go daemon tails Claude Code + codex session logs, extracts durable facts with a configured OpenAI-compatible LLM, links them into an entity graph with decay + supersession, and serves the result over CLI + HTTP (`:4141`). Single flat `main` package, SQLite storage (`~/.oracle/oracle.db`, override with `ORACLE_HOME`).

## Commands

```sh
go build -o oracle ./cmd/oracle        # build (pure-Go sqlite driver, no cgo)
go vet ./...                # no tests exist; vet is the static check
./oracle admin init         # create the db (also implicit on first use)
./oracle admin cycle --max-calls 5   # one ingest pass (max-calls caps LLM spend)
./oracle up                 # HTTP on :4141 + ingest loop every 5 min
./oracle query "..."        # hybrid search; core: ask, install, up, models, status
                            # graph namespace: entity, relations, brief, topics,
                            #   narrative, metric, merge, alias, conflicts
                            # admin namespace: cycle, enrich, reembed, sweep, repair,
                            #   referee, optimize, backup, eval/debug commands
                            # old top-level names still work as hidden aliases
go test ./...                 # unit tests (offline: temp DBs, no LLM calls)
eval/eval.sh                # retrieval eval: hit@5 over eval/probes.tsv
                            # (needs a built `oracle` on PATH + populated db)
```

Unit tests cover the pure functions and DB-level invariants (parsers, redaction, as-of predicate, supersede candidates, chunking, backup) using temp `ORACLE_HOME` dirs — they run offline. `eval/eval.sh` is the end-to-end correctness signal for retrieval changes: each line of `probes.tsv` is a query TAB expected-regex, scored hit@5 against the live db.

All remote LLM calls (extraction, judging, enrichment, ask, topics, narrative) hit one configured OpenAI-compatible chat endpoint; remote embeddings hit a configured embeddings endpoint (`internal/llm/provider.go`). Config resolution: `ORACLE_LLM_URL/KEY/MODEL` + `ORACLE_EMBED_URL/KEY/MODEL` env → `~/.oracle/config` (KEY=VALUE) → loud error with setup examples. `ORACLE_LLM_REASONING_EFFORT` is only sent when set. The local embedder (`ORACLE_LOCAL_EMBED=1`) makes remote embeddings optional.

## Architecture

Standard Go layout: a thin CLI in `cmd/oracle`, one internal package per concern; data flows left to right:

```
ingest/watch.go ─→ ingest/extract.go ─→ search/graph.go(ingest) ─→ ingest/enrich.go / search/embed.go
 tail jsonl        LLM facts +           upsert, supersede,          triples/metrics     vectors
 sessions          supersede judge       entities, decay
                                              │
store/db.go + store/schemas.go                ├─→ search/graph.go(search) / ask/ask.go / kb/kbport.go (traverse, topics, narrative)
(all under internal/)                         └─→ cmd/oracle (CLI) / serve/serve.go (HTTP + loop)
```

- **internal/ingest/watch.go** — discovers session jsonl under `~/.claude/projects` and `~/.codex/sessions`, reads from stored byte offsets, renders new lines into ~24k-char text chunks. Secrets are regex-redacted here (`redact`) before text leaves the box; extraction redacts again on the output side.
- **internal/ingest/extract.go** — chunk → facts (kind ∈ decision|fact|gotcha|preference|status|todo) via the configured LLM's JSON mode; also the supersede/contradict judge. `llm.ChatJSON` is the shared LLM helper every other file uses.
- **internal/search/graph.go** — the ingest cycle and search. Search fuses FTS/bm25 with cosine via RRF, then nudges by decayed mass, confidence, and repo match. Retrieval reinforces mass (`+0.15`, cap 3.0). Every query is logged to `traces` (training substrate for future planners).
- **internal/search/embed.go** — remote embeddings (or local ONNX) @ 512 dims, float32 blobs in `fact_vecs`, brute-force cosine (fine < 100k facts).
- **internal/kb/kbport.go / internal/ingest/enrich.go / internal/search/introspect.go** — the ported kb surface: typed entities, bitemporal aliases, entity merge, S-P-O triples + predicate registry, metric observations, n-hop traversal, k-means topics, narrative; enrichment retrofits every fact (tracked in `enrich_done`); introspection snapshot + past-ask-trajectory memory feed the reasoner.
- **internal/ask/ask.go** — multi-step tool-calling reasoner over search/entity/graph/metric, answers with `[fact-id]` citations. Note: some servers reject `reasoning_effort` alongside function tools, so `ask` never sends it.
- **internal/serve/serve.go** — HTTP mux + the 5-minute ingest loop (`oracle up`).

## Invariants — read before changing ingest or search

- **Bi-temporal, append-only.** `valid_from` is world time (when the thing became true); `recorded_at`/`superseded_at` are transaction time (when oracle learned it). Facts are never deleted or edited — supersession closes validity. `--as-of T` filters on world time: `valid_from <= T AND (no superseder OR superseder.valid_from > T)`. See `docs/facts.md` for the full model.
- **Fail loud, never advance past failure.** No fallback provider, no empty-list-on-error, no degraded semantic-off search. A failed chunk must not advance the file offset; offset advance commits atomically with its facts in one transaction (`ingestChunk`).
- **Cost is budgeted.** `--max-calls` caps extraction LLM calls per cycle; failing files get per-file exponential backoff (`error_count`/`last_error_ts`, 30 min · 2^n, cap 24 h).
- **One cycle at a time.** A lease row in `meta` (`cycle_lease`) serializes cycles across daemon + manual runs. Chunks within a file stay sequential so later facts can supersede earlier ones; files run in parallel.
- **Decay per kind.** Half-lives in `internal/search/graph.go` (`halfLifeDays`): status 7 d → preference 365 d; `mass = ε + m·2^(−age/hl)`.
- **SQLite, single writer.** `SetMaxOpenConns(1)`; WAL + 30 s busy timeout. Schema is applied idempotently at `store.OpenDB` (three blocks: `schema`, `embedSchema`, `kbSchema` — the latter tolerates duplicate-column errors, so additive ALTERs go there).
- **Entity hygiene.** All entity names lowercase-canonicalized; `kb.ValidEntityName` rejects paths/sentences; junk names drop the triple, not the batch. Merging keeps the loser's name as a live alias.

## Layout notes

- `prototype/` is the retired Python v0 — reference only, don't extend it.
- `docs/facts.md` documents the fact model; `docs/architecture.tex` the system.
- `install-daemon` branches on OS: launchd plist on macOS, systemd user unit on Linux (needs `loginctl enable-linger` to survive logout). README's install path (`/opt/homebrew/bin`) is macOS-specific.
