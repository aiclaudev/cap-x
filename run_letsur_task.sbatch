#!/bin/bash
#SBATCH --job-name=capx-letsur
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=360G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# CaP-Agent0 (M4) for ONE robosuite task with a COMMERCIAL code model via letsur gateway.
#   code  = letsur model (default gemini-3.1-pro-preview) over https://gw.letsur.ai  (NO local GPU)
#   VDM   = Qwen/Qwen3.6-35B-A3B   (local vLLM, GPU 0)   | Molmo = allenai/Molmo2-8B (local vLLM, GPU 1)
#   perception = SAM3 + ContactGraspNet + PyRoKi (local, shares GPU 0 with VDM)
#   Usage: sbatch --job-name=capx-letsur-<task> run_letsur_task.sbatch <config> [trials] [workers] [code_model]

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"; mkdir -p logs

CONFIG="${1:?usage: sbatch run_letsur_task.sbatch <config> [trials] [workers] [code_model]}"
TRIALS="${2:-30}"
WORKERS="${3:-6}"
CODE_MODEL="${4:-gemini-3.1-pro-preview}"
REASONING="${5:-low}"   # reasoning_effort for the letsur code model (low ≈ thinking off for gemini)
VDM_MODEL="Qwen/Qwen3.6-35B-A3B"
LETSUR_URL="https://gw.letsur.ai/v1/chat/completions"
LETSUR_KEY="$(cat /home/nas_main/dohyunlee/agentic-robotics-dev/.letsurkey)"
MODEL_DIR="outputs/${CODE_MODEL//\//_}"   # launch.py inserts model name (/ -> _) under outputs/

export HF_HOME=/home/nas_main/.cache/huggingface
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CONFIG=$CONFIG  TRIALS=$TRIALS  WORKERS=$WORKERS  CODE_MODEL=$CODE_MODEL (letsur)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
echo "allocated GPUs: ${G[*]}  (vdm+perc=${G[0]}  molmo=${G[1]})"

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

# ---- 1) VDM server: Qwen3.6-35B-A3B (GPU 0, shares with perception) ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CUDA_VISIBLE_DEVICES="${G[0]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model "$VDM_MODEL" --host 127.0.0.1 --port 8081 \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --reasoning-parser qwen3 \
      --gpu-memory-utilization 0.55 --max-model-len 32768 \
      --max-num-seqs 16 \
      --limit-mm-per-prompt '{"image":5}'
) > "logs/vdm_${SLURM_JOB_ID}.log" 2>&1 &
VDM_PID=$!

# ---- 2) Molmo server: allenai/Molmo2-8B (GPU 1, :8122) ----
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

echo "vdm pid $VDM_PID (:8081), molmo pid $MOLMO_PID (:8122) — waiting"
KILL_ALL="kill $VDM_PID $MOLMO_PID 2>/dev/null"
wait_ready 8081 vdm   "$VDM_PID"   || { eval "$KILL_ALL"; exit 1; }
wait_ready 8122 molmo "$MOLMO_PID" || { eval "$KILL_ALL"; exit 1; }

# ---- 3) eval: code = letsur gemini, VDM = local Qwen3.6, perception auto-launched ----
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${G[0]}"   # robosuite render + SAM3/GraspNet share the VDM GPU
export MUJOCO_EGL_DEVICE_ID=0

OUT_BASE=$(python -c "import yaml,os;print(os.path.basename(yaml.safe_load(open('$CONFIG'))['output_dir']))")
TASK_OUT="$MODEL_DIR/$OUT_BASE"
echo "task output dir: $TASK_OUT"

python capx/envs/launch.py \
    --config-path "$CONFIG" \
    --model "$CODE_MODEL" \
    --server-url "$LETSUR_URL" \
    --api-key "$LETSUR_KEY" \
    --visual-differencing-model "$VDM_MODEL" \
    --visual-differencing-model-server-url http://127.0.0.1:8081/v1/chat/completions \
    --reasoning-effort "$REASONING" \
    --total-trials "$TRIALS" --num-workers "$WORKERS" --max-tokens 16384
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down servers"
kill $VDM_PID $MOLMO_PID 2>/dev/null || true; wait 2>/dev/null || true

# ---- 4) visualization: per-task report.html + dashboard ----
if [ -d "$TASK_OUT" ]; then
  python scripts/summarize_turns.py "$TASK_OUT" || echo "!! summarize_turns failed"
  python scripts/viz_trial.py "$TASK_OUT" || echo "!! viz_trial failed"
  python scripts/viz_index.py "$MODEL_DIR" || echo "!! viz_index failed"
fi
echo "================ DONE (job $SLURM_JOB_ID, eval_rc=$EVAL_RC) ================"
exit $EVAL_RC
