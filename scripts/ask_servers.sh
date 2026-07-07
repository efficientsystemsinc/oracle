#!/bin/sh
# ask_servers.sh — serve the local-ask models via an OpenAI-compatible server.
# Two models, two processes (one model per process):
#   policy  Qwen2.5-0.5B BC tool-planner   :8398
#   synth   Qwen2.5-1.5B cited-answerer    :8397
#
# Usage: ask_servers.sh policy | synth
#
# Backend: ORACLE_ASK_BACKEND=mlx|vllm|llama, else auto-picked — mlx on
# darwin; on other platforms vllm if on PATH, else llama-server (llama.cpp,
# the right answer for CPU-only boxes). Model path defaults per backend
# (override with ORACLE_POLICY_MODEL / ORACLE_SYNTH_MODEL):
#   mlx    ~/.oracle/models/{policy,synth}_mlx      4-bit MLX dirs
#   vllm   ~/.oracle/models/{policy,synth}_merged   merged HF checkpoints
#   llama  ~/.oracle/models/{policy,synth}.gguf     GGUF files
#
# One-time model prep (merged HF checkpoints live on your training box at
# ~/extract/{policy_merged,synth_merged} — not yet on the models release):
#   mlx:   python3 -m mlx_lm convert --hf-path ~/.oracle/models/policy_merged -q --mlx-path ~/.oracle/models/policy_mlx
#   vllm:  the merged dir serves as-is
#   llama: convert_hf_to_gguf.py ~/.oracle/models/policy_merged --outfile ~/.oracle/models/policy.gguf --outtype q8_0
#
# vllm notes: both roles fit one GPU — each process takes
# --gpu-memory-utilization $ORACLE_ASK_GPU_FRAC (default 0.40). vLLM requires
# the OpenAI "model" request field, so both roles serve as name "oracle-ask"
# and the Go side must run with ORACLE_ASK_MODEL_FIELD=oracle-ask
# (mlx/llama backends ignore the field either way — leave the env unset there).
#
# Keep-alive, macOS launchd: one plist per role,
#   ProgramArguments = [.../ask_servers.sh, policy], KeepAlive = true
# Linux systemd (user unit, e.g. ~/.config/systemd/user/oracle-ask-policy.service):
#   [Service]
#   ExecStart=%h/oracle/scripts/ask_servers.sh policy
#   Restart=always
#   [Install]
#   WantedBy=default.target
set -eu

ROLE="${1:-}"
case "$ROLE" in
policy)
    PORT="${ORACLE_POLICY_PORT:-8398}"
    ;;
synth)
    PORT="${ORACLE_SYNTH_PORT:-8397}"
    ;;
*)
    echo "usage: $0 policy|synth" >&2
    exit 2
    ;;
esac

BACKEND="${ORACLE_ASK_BACKEND:-}"
if [ -z "$BACKEND" ]; then
    if [ "$(uname -s)" = "Darwin" ]; then
        BACKEND=mlx
    elif command -v vllm >/dev/null 2>&1; then
        BACKEND=vllm
    elif command -v llama-server >/dev/null 2>&1; then
        BACKEND=llama
    else
        echo "ask_servers: no backend found — install vllm or llama.cpp (llama-server), or set ORACLE_ASK_BACKEND" >&2
        exit 1
    fi
fi

model_for() { # $1 role, $2 backend
    case "$2" in
    mlx)   echo "$HOME/.oracle/models/${1}_mlx" ;;
    vllm)  echo "$HOME/.oracle/models/${1}_merged" ;;
    llama) echo "$HOME/.oracle/models/${1}.gguf" ;;
    esac
}

if [ "$ROLE" = policy ]; then
    MODEL="${ORACLE_POLICY_MODEL:-$(model_for policy "$BACKEND")}"
else
    MODEL="${ORACLE_SYNTH_MODEL:-$(model_for synth "$BACKEND")}"
fi

case "$BACKEND" in
mlx)
    if [ ! -f "$MODEL/config.json" ]; then
        echo "ask_servers: $ROLE model not found at $MODEL — run the mlx_lm convert step in this script's header" >&2
        exit 1
    fi
    exec python3 -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port "$PORT"
    ;;
vllm)
    if [ ! -f "$MODEL/config.json" ]; then
        echo "ask_servers: $ROLE model not found at $MODEL — copy the merged HF checkpoint (see header)" >&2
        exit 1
    fi
    exec vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
        --served-model-name oracle-ask \
        --gpu-memory-utilization "${ORACLE_ASK_GPU_FRAC:-0.40}" \
        --max-model-len "${ORACLE_ASK_MAX_LEN:-4096}"
    ;;
llama)
    if [ ! -f "$MODEL" ]; then
        echo "ask_servers: $ROLE model not found at $MODEL — run the gguf convert step in this script's header" >&2
        exit 1
    fi
    exec llama-server -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
        -c "${ORACLE_ASK_MAX_LEN:-4096}"
    ;;
*)
    echo "ask_servers: unknown ORACLE_ASK_BACKEND=$BACKEND (want mlx|vllm|llama)" >&2
    exit 2
    ;;
esac
