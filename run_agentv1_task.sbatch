#!/bin/bash
#SBATCH --job-name=capx-agentv1
#SBATCH --qos=own
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=360G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# agentv1 (Reflector) for ONE robosuite task. code + reflector = gemini via letsur gateway (NO local GPU).
#   reflector reuses the visual-differencing endpoint (also letsur gemini) — sends the turn VIDEO.
#   Molmo = allenai/Molmo2-8B (local vLLM, GPU 1) for the reduced-skill-lib perception API.
#   perception = SAM3 + ContactGraspNet + PyRoKi (auto-launched by the config, share GPU 0).
#   Usage: sbatch --job-name=capx-agentv1-<task> run_agentv1_task.sbatch <config> [trials] [workers] [code_model]
#   NOTE: first agentv1 run — reflector video-through-letsur is UNVERIFIED; smoke-test one task first.

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"; mkdir -p logs

CONFIG="${1:?usage: sbatch run_agentv1_task.sbatch <config> [trials] [workers] [code_model]}"
TRIALS="${2:-1}"
WORKERS="${3:-1}"
CODE_MODEL="${4:-gemini-3.1-pro-preview}"   # letsur bare name (overrides config's google/... for the gateway)
REASONING="${5:-low}"
RESUME_IDX="${6:-}"   # optional 1-based start index; appends trials range(RESUME_IDX, TRIALS+1) without overwriting
LETSUR_URL="https://gw.letsur.ai/v1/chat/completions"
KEYFILE="/home/nas_main/dohyunlee/agentic-robotics-dev/.letsurkey"   # parent repo root (same as run_letsur)
[ -s "$KEYFILE" ] || { echo "!! missing/empty $KEYFILE — provide the letsur key first"; exit 2; }
LETSUR_KEY="$(cat "$KEYFILE")"

export HF_HOME=/home/nas_main/.cache/huggingface
export HF_HUB_OFFLINE=1        # models (Molmo/SAM3) are cached — skip HF metadata pings (avoids 429 rate-limit)
export TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CONFIG=$CONFIG  TRIALS=$TRIALS  WORKERS=$WORKERS  CODE_MODEL=$CODE_MODEL (letsur, reflector=same)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
echo "allocated GPUs: ${G[*]}  (robosuite+perception=${G[0]}  molmo=${G[1]})"

PROBE="$REPO/.venv-vllm/bin/python"
wait_ready () {  # $1=port $2=name $3=pid
  for i in $(seq 1 360); do
    if "$PROBE" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$1/v1/models', timeout=3)" >/dev/null 2>&1; then
      echo "$2 ready after ~$((i*5))s"; return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then echo "!! $2 died"; tail -50 "logs/$2_${SLURM_JOB_ID}.log"; return 1; fi
    sleep 5
  done
  echo "!! $2 not ready in time"; tail -50 "logs/$2_${SLURM_JOB_ID}.log"; return 1
}

# ---- 1) Molmo server: allenai/Molmo2-8B (GPU 1, :8122) — perception for the reduced-skill-lib API ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CUDA_VISIBLE_DEVICES="${G[1]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model allenai/Molmo2-8B --host 127.0.0.1 --port 8122 \
      --trust-remote-code \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --gpu-memory-utilization 0.45 --max-model-len 8192 \
      --limit-mm-per-prompt '{"image":1}'
) > "logs/molmo_${SLURM_JOB_ID}.log" 2>&1 &
MOLMO_PID=$!
echo "molmo pid $MOLMO_PID (:8122) — waiting"
wait_ready 8122 molmo "$MOLMO_PID" || { kill $MOLMO_PID 2>/dev/null; exit 1; }

# ---- 2) eval: code + reflector + initial-scene = gemini via letsur; perception auto-launched ----
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${G[0]}"   # robosuite render + SAM3/GraspNet share GPU 0
export MUJOCO_EGL_DEVICE_ID=0

TASK_OUT=$(python -c "import yaml; c=yaml.safe_load(open('$CONFIG')); v=c.get('version') or 'cap-agent0'; p=c['output_dir'].split('/'); p.insert(-1,'$CODE_MODEL'.replace('/','_')); p.insert(-2,v); print('/'.join(p))")
MODEL_DIR=$(dirname "$TASK_OUT")   # outputs/<version>/<model>
echo "task output dir: $TASK_OUT"

RESUME_ARG=()
[ -n "$RESUME_IDX" ] && RESUME_ARG=(--resume-idx "$RESUME_IDX")
echo "RESUME_IDX=${RESUME_IDX:-<none>}  (trials range: ${RESUME_IDX:-1}..$TRIALS)"

python capx/envs/launch.py \
    --config-path "$CONFIG" \
    --model "$CODE_MODEL" \
    --server-url "$LETSUR_URL" \
    --api-key "$LETSUR_KEY" \
    --reflector-model "$CODE_MODEL" \
    --visual-differencing-model "$CODE_MODEL" \
    --visual-differencing-model-server-url "$LETSUR_URL" \
    --visual-differencing-model-api-key "$LETSUR_KEY" \
    --reasoning-effort "$REASONING" \
    "${RESUME_ARG[@]}" \
    --total-trials "$TRIALS" --num-workers "$WORKERS" --max-tokens 16384
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down molmo"
kill $MOLMO_PID 2>/dev/null || true; wait 2>/dev/null || true

# ---- 3) visualization: per-task report.html + dashboard index.html ----
if [ -d "$TASK_OUT" ]; then
  python scripts/summarize_turns.py "$TASK_OUT" || echo "!! summarize_turns failed"
  python scripts/viz_trial.py "$TASK_OUT" || echo "!! viz_trial failed"
  python scripts/viz_index.py "$MODEL_DIR" || echo "!! viz_index failed"
fi
echo "================ DONE (job $SLURM_JOB_ID, eval_rc=$EVAL_RC) ================"
exit $EVAL_RC
