#!/bin/bash
#SBATCH --job-name=capx-agent0-vl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=220G
#SBATCH --time=01:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# CaP-Agent0-ish run, fully local (no API key):
#   code+VDM = Qwen2.5-VL-7B-Instruct (vLLM)  ·  perception = SAM3 + ContactGraspNet + PyRoKi
#   config = non-privileged + multiturn + VDM(use_img_differencing) + skill-library API
# One GPU shared by the VL server and the perception/sim stack.

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"
mkdir -p logs
export HF_HOME=/home/nas_main/.cache/huggingface   # SHARED cache holds gated SAM3 (3.3G) + Qwen2.5-VL
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true

# ---------------------------------------------------------------------------
# 1) Serve Qwen2.5-VL via vLLM (.venv-vllm) — code generation AND VDM
# ---------------------------------------------------------------------------
VLLM_LOG="$REPO/logs/vllm_vl_${SLURM_JOB_ID}.log"
(
  source "$REPO/.venv-vllm/bin/activate"
  exec python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-VL-7B-Instruct \
      --host 127.0.0.1 --port 8080 \
      --dtype bfloat16 \
      --tensor-parallel-size 1 \
      --gpu-memory-utilization 0.35 \
      --max-model-len 32768 \
      --limit-mm-per-prompt '{"image":5}'
) > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "vLLM (Qwen2.5-VL) serving pid $VLLM_PID, log $VLLM_LOG"
PROBE="$REPO/.venv-vllm/bin/python"
READY=0
for i in $(seq 1 300); do   # up to 25 min (VL load)
  if "$PROBE" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=3)" >/dev/null 2>&1; then
    echo "vLLM ready after ~$((i*5))s"; READY=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "!! vLLM died. log tail:"; tail -60 "$VLLM_LOG"; exit 1; fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "!! vLLM not ready in time"; tail -60 "$VLLM_LOG"; kill "$VLLM_PID" 2>/dev/null; exit 1; }

# ---------------------------------------------------------------------------
# 2) Run CaP-Agent0 eval (robosuite .venv; config auto-launches SAM3/GraspNet/PyRoKi)
# ---------------------------------------------------------------------------
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface   # re-assert after activate

python capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --server-url http://127.0.0.1:8080/v1/chat/completions \
    --visual-differencing-model Qwen/Qwen2.5-VL-7B-Instruct \
    --visual-differencing-model-server-url http://127.0.0.1:8080/v1/chat/completions \
    --total-trials 1 --num-workers 1 --max-tokens 4096
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down vLLM"
kill "$VLLM_PID" 2>/dev/null || true
wait "$VLLM_PID" 2>/dev/null || true
echo "================ DONE (job $SLURM_JOB_ID) ================"
exit $EVAL_RC
