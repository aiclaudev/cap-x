#!/bin/bash
#SBATCH --job-name=capx-perc-coder
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# Perception-grounded run to SEE perception results:
#   code = Qwen2.5-Coder-7B (valid syntax + correct API names)
#   perception = SAM3 + ContactGraspNet + PyRoKi (high-level FrankaControlApi)
#   config = non-privileged + multiturn (NO VDM)  ->  get_object_pose logs SAM3 segmentation
# The trial.py hook persists those perception images to perception/trial_NN/ for the viz tool.

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"
mkdir -p logs
export HF_HOME=/home/nas_main/.cache/huggingface   # SHARED cache holds gated SAM3 (3.3G)
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true

# ---- 1) Serve Qwen2.5-Coder via vLLM ----
VLLM_LOG="$REPO/logs/vllm_coder_${SLURM_JOB_ID}.log"
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-Coder-7B-Instruct \
      --host 127.0.0.1 --port 8080 \
      --dtype bfloat16 --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.40 --max-model-len 32768
) > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM (Coder) pid $VLLM_PID, log $VLLM_LOG"
PROBE="$REPO/.venv-vllm/bin/python"
READY=0
for i in $(seq 1 240); do
  if "$PROBE" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=3)" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i*5))s"; READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "!! vLLM died"; tail -60 "$VLLM_LOG"; exit 1; fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "!! vLLM not ready"; tail -60 "$VLLM_LOG"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

# ---- 2) Run eval (robosuite .venv; config auto-launches SAM3/GraspNet/PyRoKi) ----
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface

python capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn.yaml \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --server-url http://127.0.0.1:8080/v1/chat/completions \
    --total-trials 1 --num-workers 1 --max-tokens 4096
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down vLLM"
kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true
echo "================ DONE (job $SLURM_JOB_ID) ================"
exit $EVAL_RC
