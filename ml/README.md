# ml/ — oracle's small-model program

Every model follows the same loop: **frontier-LLM teacher → synthetic/derived data →
small model trained on cloud (TPU v6e / A100) → gated eval → MLX Metal
inference inside the binary.** Ceilings so far were always data distribution,
never model size.

## Deployed (fetch weights: `models/fetch.sh`)

| model | base | result | serving |
|---|---|---|---|
| judge (write-path supersession) | bert-base 110M | 98.3% clean 3-class; OOD: temporal 81% bin, repo 81% bin, cross-generator 97.8% | ONNX/MLX, `ORACLE_LOCAL_JUDGE=shadow|active`, margin-gated (`ORACLE_JUDGE_MARGIN`) |
| embedder (query+corpus vectors) | e5-base 110M | **beats Azure text-embedding-3-large on-domain: 81.5 vs 79.7 hit@5** | ONNX/MLX, `ORACLE_LOCAL_EMBED=1` + `oracle reembed` |
| MLX engine | — | 10–19x over ORT-CPU (embed 11ms, judge 15ms on M4 Max) | `ORACLE_MLX=1`, cpp/oraclemlx |

## In training / staged

- **extract/** — chunk→facts distillation (Qwen2.5 1.5B vs 0.5B LoRA, A100 A/B in flight); gate = teacher-match on 150 held-out chunks; serve via mlx-lm convert behind `ORACLE_LOCAL_EXTRACT`
- **policy/** — read-policy BC on ask trajectories (data accumulating)

## House rules

- Teacher = the deployed prompt verbatim (extract_prompt.txt is pulled from extract.go), so distillation targets production behavior.
- Every model ships with an ID *and* OOD eval before a flag exists for it.
- transformers==4.46.x for anything Flax; py3.13 venv (3.14 breaks tokenizers builds).
- Long cloud jobs get a box-side cron supervisor; nohup alone dies with the ssh session.
- Never scp over a file a process has open.
