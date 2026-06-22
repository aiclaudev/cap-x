#!/bin/bash
#SBATCH --job-name=capx-agent0-full
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# Full local CaP-Agent0 (no API key):
#   code  = Qwen/Qwen3-Coder-Next      (vLLM, TP=2, GPUs 0+1)        non-thinking code model
#   VDM   = Qwen/Qwen3.6-35B-A3B        (vLLM, GPU 2, reasoning-parser qwen3, VLM)
#   perception = SAM3 + ContactGraspNet + PyRoKi   (shares GPU 2 with the VDM server)
#   config = cube_stack_multiturn_vdm_reduced_api_skill_lib  (multiturn + VDM + skill-library)

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"; mkdir -p logs
export HF_HOME=/home/nas_main/.cache/huggingface   # shared cache: gated SAM3 + both Qwen3 models
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
echo "allocated GPUs: ${G[*]}  (coder=${G[0]},${G[1]}  vdm+perc=${G[2]})"

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

# ---- 1) code server: Qwen3-Coder-Next (TP=2 on GPUs 0,1) ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"   # FlashInfer JIT needs CUDA headers
  export CUDA_VISIBLE_DEVICES="${G[0]},${G[1]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3-Coder-Next --host 127.0.0.1 --port 8080 \
      --tensor-parallel-size 2 --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len 32768
) > "logs/coder_${SLURM_JOB_ID}.log" 2>&1 &
CODER_PID=$!

# ---- 2) VDM server: Qwen3.6-35B-A3B (GPU 2, leave room for perception) ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"   # FlashInfer JIT needs CUDA headers
  export CUDA_VISIBLE_DEVICES="${G[2]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen3.6-35B-A3B --host 127.0.0.1 --port 8081 \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --reasoning-parser qwen3 \
      --gpu-memory-utilization 0.55 --max-model-len 32768 \
      --max-num-seqs 16 \
      --limit-mm-per-prompt '{"image":5}'
) > "logs/vdm_${SLURM_JOB_ID}.log" 2>&1 &
VDM_PID=$!

echo "coder pid $CODER_PID (:8080), vdm pid $VDM_PID (:8081) — waiting for both"
wait_ready 8080 coder "$CODER_PID" || { kill $CODER_PID $VDM_PID 2>/dev/null; exit 1; }
wait_ready 8081 vdm   "$VDM_PID"   || { kill $CODER_PID $VDM_PID 2>/dev/null; exit 1; }

# ---- 3) eval: code=Coder-Next, VDM=35B-A3B, perception auto-launched by config ----
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${G[2]}"   # robosuite render + SAM3/GraspNet share the VDM GPU
export MUJOCO_EGL_DEVICE_ID=0

python capx/envs/launch.py \
    --config-path env_configs/cube_stack/franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml \
    --model Qwen/Qwen3-Coder-Next \
    --server-url http://127.0.0.1:8080/v1/chat/completions \
    --visual-differencing-model Qwen/Qwen3.6-35B-A3B \
    --visual-differencing-model-server-url http://127.0.0.1:8081/v1/chat/completions \
    --total-trials 1 --num-workers 1 --max-tokens 8192
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down servers"
kill $CODER_PID $VDM_PID 2>/dev/null || true; wait 2>/dev/null || true
echo "================ DONE (job $SLURM_JOB_ID) ================"
exit $EVAL_RC
