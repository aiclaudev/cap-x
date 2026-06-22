#!/bin/bash
#SBATCH --job-name=capx-qwen-mt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=192G
#SBATCH --time=00:40:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# CaP-X first local-vLLM run: serve Qwen2.5-Coder-7B-Instruct, then run a
# privileged + multi-turn eval (2 trials) against it. One GPU, shared between
# the vLLM server (cu130 venv) and the robosuite sim (cu128 venv).

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"
mkdir -p logs
export HF_HOME=/home/nas_main/dohyunlee/.cache/huggingface
unset PYTHONPATH                  # don't let the NGC image's site-packages leak into our venvs
export GIT_PYTHON_REFRESH=quiet   # belt-and-suspenders for GitPython import

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

# ---------------------------------------------------------------------------
# 1) Serve Qwen via vLLM (.venv-vllm)  — OpenAI-compatible API on :8080
# ---------------------------------------------------------------------------
VLLM_LOG="$REPO/logs/vllm_${SLURM_JOB_ID}.log"
(
  source "$REPO/.venv-vllm/bin/activate"
  exec python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 127.0.0.1 --port 8080 \
      --dtype bfloat16 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.5 \
      --max-model-len 32768
) > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "vLLM serving (pid $VLLM_PID), log: $VLLM_LOG"
echo "Waiting for model load + server ready (first run downloads ~15GB) ..."
READY=0
PROBE="$REPO/.venv-vllm/bin/python"   # curl may be absent in the container; use stdlib urllib
for i in $(seq 1 240); do   # up to 20 min
  if "$PROBE" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=3)" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i*5))s"; READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "!! vLLM process died before becoming ready. Tail of log:"; tail -60 "$VLLM_LOG"; exit 1
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "!! vLLM did not become ready in time. Tail of log:"; tail -60 "$VLLM_LOG"
  kill "$VLLM_PID" 2>/dev/null; exit 1
fi

# ---------------------------------------------------------------------------
# 2) Run the eval (robosuite .venv via capx_activate.sh)
# ---------------------------------------------------------------------------
source "$REPO/capx_activate.sh"   # activates .venv + CUDA_HOME + GL loaders + MUJOCO_GL=egl

python capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_privileged_multiturn.yaml \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --server-url http://127.0.0.1:8080/v1/chat/completions \
    --max-tokens 8192
EVAL_RC=$?

# ---------------------------------------------------------------------------
# 3) Cleanup
# ---------------------------------------------------------------------------
echo "eval exit code: $EVAL_RC ; shutting down vLLM"
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "================ DONE (job $SLURM_JOB_ID) ================"
exit $EVAL_RC
