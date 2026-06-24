#!/bin/bash
#SBATCH --job-name=capx-a0
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=14
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# Parameterized CaP-Agent0 (M4) run for ONE robosuite task.
#   Usage:  sbatch --job-name=capx-a0-<task> run_agent0_task.sbatch <config-path> [trials] [workers]
#   code  = Qwen/Qwen3-Coder-Next  (TP=2, GPUs 0+1) | VDM = Qwen/Qwen3.6-35B-A3B (GPU 2)
#   Molmo = allenai/Molmo2-8B (GPU 3, :8122)         | perception = SAM3+GraspNet+PyRoKi (GPU 2)
#   On finish: writes <task>/report.html (viz_trial) + the top-level index.html (viz_index).

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"; mkdir -p logs

CONFIG="${1:?usage: sbatch run_agent0_task.sbatch <config-path> [trials] [workers]}"
TRIALS="${2:-30}"
WORKERS="${3:-6}"
MODEL="Qwen/Qwen3-Coder-Next"
VDM_MODEL="Qwen/Qwen3.6-35B-A3B"
MODEL_DIR="outputs/${MODEL//\//_}"   # launch.py inserts the model name (/ -> _) under outputs/

export HF_HOME=/home/nas_main/.cache/huggingface
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "CONFIG=$CONFIG  TRIALS=$TRIALS  WORKERS=$WORKERS"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L || true
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
echo "allocated GPUs: ${G[*]}  (coder=${G[0]},${G[1]}  vdm+perc=${G[2]}  molmo=${G[3]})"

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
      --model "$MODEL" --host 127.0.0.1 --port 8080 \
      --tensor-parallel-size 2 --dtype bfloat16 \
      --gpu-memory-utilization 0.90 --max-model-len 32768
) > "logs/coder_${SLURM_JOB_ID}.log" 2>&1 &
CODER_PID=$!

# ---- 2) VDM server: Qwen3.6-35B-A3B (GPU 2, leave room for perception) ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CUDA_VISIBLE_DEVICES="${G[2]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model "$VDM_MODEL" --host 127.0.0.1 --port 8081 \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --reasoning-parser qwen3 \
      --gpu-memory-utilization 0.55 --max-model-len 32768 \
      --max-num-seqs 16 \
      --limit-mm-per-prompt '{"image":5}'
) > "logs/vdm_${SLURM_JOB_ID}.log" 2>&1 &
VDM_PID=$!

# ---- 2b) Molmo server: allenai/Molmo2-8B (GPU 3, :8122 — point_prompt_molmo backend) ----
(
  source "$REPO/.venv-vllm/bin/activate"
  export CUDA_HOME=/usr/local/cuda; export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
  export CUDA_VISIBLE_DEVICES="${G[3]}"
  exec python -m vllm.entrypoints.openai.api_server \
      --model allenai/Molmo2-8B --host 127.0.0.1 --port 8122 \
      --trust-remote-code \
      --tensor-parallel-size 1 --dtype bfloat16 \
      --gpu-memory-utilization 0.45 --max-model-len 8192 \
      --limit-mm-per-prompt '{"image":1}'
) > "logs/molmo_${SLURM_JOB_ID}.log" 2>&1 &
MOLMO_PID=$!

echo "coder pid $CODER_PID (:8080), vdm pid $VDM_PID (:8081), molmo pid $MOLMO_PID (:8122) — waiting for all"
KILL_ALL="kill $CODER_PID $VDM_PID $MOLMO_PID 2>/dev/null"
wait_ready 8080 coder "$CODER_PID" || { eval "$KILL_ALL"; exit 1; }
wait_ready 8081 vdm   "$VDM_PID"   || { eval "$KILL_ALL"; exit 1; }
wait_ready 8122 molmo "$MOLMO_PID" || { eval "$KILL_ALL"; exit 1; }

# ---- 3) eval ----
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface
export CUDA_VISIBLE_DEVICES="${G[2]}"   # robosuite render + SAM3/GraspNet share the VDM GPU
export MUJOCO_EGL_DEVICE_ID=0

# task output dir = outputs/<model>/<config's output_dir basename>
OUT_BASE=$(python -c "import yaml,os;print(os.path.basename(yaml.safe_load(open('$CONFIG'))['output_dir']))")
TASK_OUT="$MODEL_DIR/$OUT_BASE"
echo "task output dir: $TASK_OUT"

python capx/envs/launch.py \
    --config-path "$CONFIG" \
    --model "$MODEL" \
    --server-url http://127.0.0.1:8080/v1/chat/completions \
    --visual-differencing-model "$VDM_MODEL" \
    --visual-differencing-model-server-url http://127.0.0.1:8081/v1/chat/completions \
    --total-trials "$TRIALS" --num-workers "$WORKERS" --max-tokens 8192
EVAL_RC=$?

echo "eval exit $EVAL_RC ; shutting down servers"
kill $CODER_PID $VDM_PID $MOLMO_PID 2>/dev/null || true; wait 2>/dev/null || true

# ---- 4) visualization: per-task report.html + top-level dashboard ----
if [ -d "$TASK_OUT" ]; then
  echo "generating report.html for $TASK_OUT"
  python scripts/summarize_turns.py "$TASK_OUT" || echo "!! summarize_turns failed"
  python scripts/viz_trial.py "$TASK_OUT" || echo "!! viz_trial failed"
  echo "generating dashboard index.html for $MODEL_DIR"
  python scripts/viz_index.py "$MODEL_DIR" || echo "!! viz_index failed"
else
  echo "!! task output dir not found, skipping viz: $TASK_OUT"
fi

echo "================ DONE (job $SLURM_JOB_ID, eval_rc=$EVAL_RC) ================"
exit $EVAL_RC
