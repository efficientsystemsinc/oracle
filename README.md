# oracle

Bi-temporal fact graph over every agent session on this machine. A Go daemon
watches Claude Code + codex session logs, extracts durable facts with any OpenAI-compatible LLM,
links them into an entity graph with decay + supersession, and serves the
result over HTTP + CLI. Coding agents query it for perfect context.


## Ask it anything about your own history

`ask` is the heart of oracle: a multi-hop reasoner that plans retrieval over the
graph — reformulating searches, following entities, comparing time periods —
then answers with citations into your own facts.

```
$ oracle ask "why did we switch the prod database, and what broke during the migration?"

The database moved because connection pooling collapsed under the indexer's
write bursts [4821]. The migration hit one failure: prepared-statement caching
had to be disabled behind the transaction pooler [4903] — as of Jun 16 that
setting is required in every client [5122].

confidence: 0.87
-- sources --
[4821] (decision, api, 32d) The backing store moved to managed Postgres because...
[4903] (gotcha,   api, 30d) Clients behind the pooler must disable statement caching...
[5122] (fact,     api,  9d) All services now set statement_cache_size=0...
```

Every claim carries a fact id you can inspect (`oracle graph entity <name>`,
`GET /facts/{id}`). Answers are **calibrated**: a confidence score is computed
from citation coverage and retrieval strength, and below threshold oracle
prefixes the answer with `LOW CONFIDENCE` instead of guessing — on a blind
holdout it produced zero confidently-wrong answers.

Time travel works here too: `oracle ask --as-of 2026-06-01 "..."` answers from
what was true *then*, using facts that have since been superseded.

An experimental fully-local mode (`ORACLE_ASK_LOCAL=1`) runs the entire loop —
planning and answer synthesis — on bundled small models with zero remote
calls, at about 3x the speed and ~70% of the answer quality of the remote loop.

## Ask your own history

`ask` is the heart of oracle: a multi-hop reasoner that plans retrieval over
the graph — reformulating searches, following entities, comparing time
periods — and answers with citations into your own facts.

```
$ oracle ask "why did we switch the prod database, and what broke?"

The database moved because connection pooling collapsed under the indexer's
write bursts [4821]. One gotcha survived the migration: prepared-statement
caching must stay disabled behind the transaction pooler [4903] — as of
Jun 16 every client sets it explicitly [5122].

confidence: 0.87
```

Every claim cites a fact id you can inspect. Confidence is calibrated from
citation coverage and retrieval strength — below threshold, oracle says
LOW CONFIDENCE instead of guessing (zero confidently-wrong answers on a blind
holdout). `--as-of` answers from what was true at any past date. `--json`
returns `{answer, confidence, sources[]}` for agents that want both the
answer and the underlying facts. An experimental `ORACLE_ASK_LOCAL=1` mode
runs the whole loop on bundled small models with zero remote calls.

## Install

```sh
curl -fsSL https://raw.githubusercontent.com/efficientsystemsinc/oracle/main/install.sh | sh
```

downloads the latest release binary (darwin-arm64 / linux-amd64), installs it
to `/usr/local/bin` or `~/.local/bin`, and prints the model-pull + daemon
setup steps. Or build from source:

## Run

```sh
go build -o oracle ./cmd/oracle && install oracle ~/.local/bin/   # or /opt/homebrew/bin on macOS
oracle init              # create ~/.oracle/oracle.db
oracle cycle             # one ingest pass (--max-calls N, --since-days N)
oracle up                # HTTP on :4141 + ingest loop every 5 min
oracle install-daemon    # launchd (macOS) / systemd --user (Linux): keep `oracle up` alive
oracle backup            # snapshot ~/.oracle/oracle.db (cycle also does this daily)
```

`oracle help` lists every command with a one-line summary; `oracle <cmd> -h`
shows flags.

## Query

```sh
oracle query "pgbouncer statement cache"        # decay+bm25 ranked facts
oracle query --as-of 2026-06-20 "prod topology" # state of knowledge back then
oracle brief --repo quasar                      # standing brief per repo
oracle entity pgbouncer                         # entity view + co-mentions
oracle graph pgbouncer --hops 2                 # typed S-P-O traversal
oracle metric pass_at_1                         # numeric time series (kb port)
oracle narrative quasar                         # LLM chronology of a subject
oracle topics -k 12                             # embedding clusters, LLM-labelled
oracle merge quasar quasar-api-gateway          # entity merge (loser -> alias)
oracle enrich --max-calls 50                    # retrofit triples/types/metrics
oracle status
```

HTTP: `GET /health /query?q= /brief?repo= /entity/{name} /facts/{id}`, `POST /cycle`.

Agent skill: `cp -r skills/oracle ~/.claude/skills/` — teaches Claude Code
sessions to pull briefs/facts from the daemon instead of re-asking the user.

## Model

- **facts** — atomic standalone statements, kind ∈ decision|fact|gotcha|preference|status|todo.
  Bi-temporal: `valid_from` (world time) vs `recorded_at`/`superseded_at`
  (transaction time). Nothing is deleted; supersession closes validity.
- **decay** — retrieval mass halves per kind-specific half-life (status 7d …
  preference 365d) and is reinforced on retrieval. `mass = ε + m·2^(-age/hl)`.
- **entities** — canonical named systems, linked to facts, with co-mention
  edges. The `entities` FTS column + repo boost make search entity-aware.
- **supersession** — new facts are judged against their FTS nearest neighbours
  (same repo+kind); superseded facts drop out of default search but remain
  queryable via `--as-of`.
- **traces** — every query + result set is logged: training substrate for the
  MuZero-style read/write planners that come next.

Extraction runs on any OpenAI-compatible chat endpoint. Configure via env or
`~/.oracle/config` (KEY=VALUE lines):

```sh
# OpenAI
ORACLE_LLM_URL=https://api.openai.com/v1/chat/completions
ORACLE_LLM_KEY=sk-...
ORACLE_LLM_MODEL=gpt-4.1

# Ollama (fully local, no key)
ORACLE_LLM_URL=http://localhost:11434/v1/chat/completions
ORACLE_LLM_MODEL=llama3.1

# Azure OpenAI (model is in the deployment URL)
ORACLE_LLM_URL=https://<resource>.openai.azure.com/openai/deployments/<dep>/chat/completions?api-version=2025-01-01-preview
ORACLE_LLM_KEY=<key>
```

Remote embeddings use the same pattern (`ORACLE_EMBED_URL` / `ORACLE_EMBED_KEY`
/ `ORACLE_EMBED_MODEL`) — or skip them entirely with the bundled local
embedder (`oracle models pull` + `ORACLE_LOCAL_EMBED=1`). Search and judging
run on local models; the remote LLM is only needed for extraction and `ask`.
Failures are loud; a failed chunk never advances the file offset. Secrets are
regex-redacted before leaving the box.

⚠️ The fact graph is a distillation of your coding sessions — treat
`~/.oracle/` as sensitive (see SECURITY.md).

`prototype/` holds the retired Python v0 — reference only.

## The models

oracle ships its own models — small, purpose-trained, and run entirely on your
machine (MLX on Apple silicon, ONNX Runtime everywhere else). This is the core
design bet: memory infrastructure should not rent intelligence per query.

| model | size | job | how it runs |
|---|---|---|---|
| **embedder** | 110M | turns every query and fact into vectors for semantic search — beat a leading hosted embedding API on-domain (81.5 vs 79.7 hit@5) | in-process, ~11ms on Metal; powers every `query` |
| **judge** | 110M | the write-path referee: does a new fact *supersede*, *contradict*, or *coexist with* an old one — the mechanism behind time travel | shadow mode by default (`ORACLE_LOCAL_JUDGE=shadow\|active`, margin-gated escalation to your LLM) |
| **ask policy** | 0.5B | plans multi-hop retrieval for `ask`: which searches to run, which entities to follow, when to stop | `ORACLE_ASK_LOCAL=1`, 4-bit via mlx_lm server |
| **ask synthesis** | 1.5B | writes the final cited answer from gathered facts | same local mode — zero remote calls end to end |
| **MLX runtime** | — | hand-built Metal engine (full bert forward, fp16, compiled/fused) + GPU-resident vector store: 10–19x over CPU inference, whole-query p50 ~11ms | automatic with `ORACLE_MLX=1` on Apple silicon |

All were trained by distillation on this system's own data: the deployed
extraction prompt is the teacher, synthetic hard cases fill distribution gaps,
and every model passed in- and out-of-distribution eval gates before earning a
flag (the ones that didn't pass ship default-off, with their numbers in
`docs/history.md` — the gates are public too).

```sh
oracle models pull    # fetch all weights (release-pinned, sha256-verified) + onnxruntime
oracle models         # what's present vs missing
```

Your extraction LLM (the one thing you bring) improves the graph; the graph
generates training data; the models get smaller and closer to the metal. That
loop is the roadmap.


## MLX inference engine (Apple Metal)

`cpp/oraclemlx/` is a C++17 MLX engine for the two local models (judge_v2,
embedder_v3): full bert-base forward in MLX ops, fp16 weights/compute with
fp32 layer-norm/softmax/heads, `mx::compile`d. ~10-19x faster than the ONNX
Runtime CPU path on an M4 Max (embed single 11ms vs 121ms, batch-16 104ms vs
1936ms; judge single 15ms vs 241ms).

Build (MLX pinned at v0.30.6):

```sh
# 1. weights: export fp32 safetensors straight from the ONNX graphs
python3.13 -m venv scripts/.venv-mlx
scripts/.venv-mlx/bin/pip install onnx safetensors numpy mlx==0.30.6
scripts/.venv-mlx/bin/python scripts/export_safetensors.py

# 2. dylib — two modes:
#    full Xcode present: FetchContent source build
cmake -S cpp/oraclemlx -B cpp/oraclemlx/build && cmake --build cpp/oraclemlx/build -j
#    Command-Line-Tools-only box (no Metal offline compiler): link the pip wheel
cmake -S cpp/oraclemlx -B cpp/oraclemlx/build \
  -DOMLX_MLX_ROOT=$(scripts/.venv-mlx/bin/python -c 'import mlx,os;print(os.path.dirname(mlx.__file__))')
cmake --build cpp/oraclemlx/build -j

# 3. install next to the models
mkdir -p ~/.oracle/models/lib
cp cpp/oraclemlx/build/{liboraclemlx.dylib,libmlx.dylib,mlx.metallib} ~/.oracle/models/lib/
```

Env vars: `ORACLE_MLX=1` routes `embedLocal`/`judgeLocal` to MLX (missing
dylib/weights is then a loud error — no silent ORT fallback; unset to use ORT
explicitly). `ORACLE_MLX_DYLIB=/path/liboraclemlx.dylib` overrides discovery
(default: `~/.oracle/models/lib/`, then `cpp/oraclemlx/build/`). Parity gate:
`go test ./internal/infer -run TestMLX -v` (embed cosine >= 0.999, judge argmax + <5e-2).
