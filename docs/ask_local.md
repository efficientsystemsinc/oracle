# ask — local policy + synth loop (default path)

`ORACLE_ASK_LOCAL=1 oracle ask` runs fully local: a BC-trained policy model
plans the tool sequence, oracle executes the tools against the fact graph,
and a trained synthesis model writes the cited answer. Zero frontier
(gpt-5.5) calls. It is OPT-IN (default stays the classic 8-round gpt-5.5
loop) because the 2026-07-07 gate failed on quality — see below.

## Models

| role   | base          | serves on | mlx path                     | trained on |
|--------|---------------|-----------|------------------------------|------------|
| policy | Qwen2.5-0.5B  | :8398     | `~/.oracle/models/policy_mlx`| question → tool lines (`search(q)`/`entity(n)`/`graph(n)`/`metric(n)`/`STOP`) |
| synth  | Qwen2.5-1.5B  | :8397     | `~/.oracle/models/synth_mlx` | question+facts → cited `[id]` answer, ask conventions |

Merged HF checkpoints live on your training box (`~/extract/{policy_merged,synth_merged}`)
(zone us-central1-b). Setup:

```sh
mkdir -p ~/.oracle/models/{policy_merged,synth_merged}
scp -r '<train-box>:~/extract/policy_merged/*' ~/.oracle/models/policy_merged/
scp -r '<train-box>:~/extract/synth_merged/*'  ~/.oracle/models/synth_merged/
python3 -m mlx_lm convert --hf-path ~/.oracle/models/policy_merged -q --mlx-path ~/.oracle/models/policy_mlx
python3 -m mlx_lm convert --hf-path ~/.oracle/models/synth_merged  -q --mlx-path ~/.oracle/models/synth_mlx
```

## Serving

Two OpenAI-compatible server processes (one model per process), via one
wrapper that picks the backend per platform — `mlx_lm` on macOS, `vllm` or
`llama-server` (llama.cpp; the CPU-only answer) elsewhere, forced with
`ORACLE_ASK_BACKEND=mlx|vllm|llama`:

```sh
scripts/ask_servers.sh policy   # :8398
scripts/ask_servers.sh synth    # :8397
```

Non-mlx model prep: vllm serves the merged HF dirs
(`~/.oracle/models/{policy,synth}_merged`) as-is; llama.cpp needs a GGUF
conversion (`convert_hf_to_gguf.py … --outtype q8_0`). vLLM requires the
OpenAI `model` request field — run oracle with
`ORACLE_ASK_MODEL_FIELD=oracle-ask` for that backend only.

Keep-alive: launchd plist per role on macOS (`ProgramArguments =
[.../ask_servers.sh, policy]`, `KeepAlive = true`); systemd user unit per role
on Linux (see the script header). Overrides: `ORACLE_POLICY_MODEL/PORT`,
`ORACLE_SYNTH_MODEL/PORT`; the Go side honors `ORACLE_POLICY_URL` /
`ORACLE_SYNTH_URL`.

If a server is down, `ask` fails loudly — it never silently falls back to the
remote loop.

## Loop shape (internal/ask/asklocal.go)

1. one policy call → tool lines (tolerant parse: `STOP()`/dupes ok; cap 4, dedupe)
2. execute via `runTool` (same machinery as classic), accumulate facts
3. widening: if retrieval strength < 0.5 (askConfidence's `ret` feature) or
   nothing surfaced, ONE alternate policy plan ("previous attempt found
   nothing useful") is executed too
4. one synth call: question + top-15-by-score facts → cited answer
5. `askConfidence` + abstain + cited-only reinforcement — identical to classic

## Eval gate

`oracle askab -n 40` — samples answerable probes from `eval/probes_1k.tsv`,
runs classic vs local, reports regex-hit rate (probe regex vs cited facts /
answer), p50 wall-clock, and frontier chat-completion counts (local must
be 0). Gate: local >= 85% of classic's hit rate.

2026-07-07 run (n=40, seed 42):

| arm     | regex-hit      | p50   | errors | gpt-5.5 calls |
|---------|----------------|-------|--------|---------------|
| classic | 31/40 (77.5%)  | 19.0s | 0      | 316*          |
| local   | 22/40 (55.0%)  |  6.9s | 0      | 0             |

*classic count from the pre-fix counter which also included Azure embedding
calls; chat rounds alone are ~5-6/probe.

Verdict: local wins latency 2.8x and cost (zero frontier calls) but is at
71% of classic's hit rate — below the 85% gate, so it stays behind
ORACLE_ASK_LOCAL=1. Rerun `oracle askab -n 40` after any policy/synth
retrain; flip useLocalAsk() when it passes.
