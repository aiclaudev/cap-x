#!/bin/bash
#SBATCH --job-name=capx-sam3-replay
#SBATCH --qos=own
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=180G
#SBATCH --time=00:40:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# Offline re-render of SAM3 overlays (confidence chips) — NO sim / NO LLM.
# Starts a SAM3 server (GPU 0, :8114), then runs scripts/replay_sam3_overlays.py.
#   Usage: sbatch run_sam3_replay.sbatch <perception_root> <trials>   e.g. "1-10"

set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"; mkdir -p logs

PERCEPTION_ROOT="${1:?usage: sbatch run_sam3_replay.sbatch <perception_root> <trials>}"
TRIALS="${2:?usage: sbatch run_sam3_replay.sbatch <perception_root> <trials>}"

export HF_HOME=/home/nas_main/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
export GIT_PYTHON_REFRESH=quiet

echo "================ job $SLURM_JOB_ID on $(hostname) ================"
echo "PERCEPTION_ROOT=$PERCEPTION_ROOT  TRIALS=$TRIALS  CVD=$CUDA_VISIBLE_DEVICES"

source "$REPO/capx_activate.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"   # first allocated GPU

# ---- SAM3 server (:8114) ----
( exec python capx/serving/launch_sam3_server.py --device cuda --port 8114 ) \
    > "logs/sam3_${SLURM_JOB_ID}.log" 2>&1 &
SAM3_PID=$!
echo "sam3 pid $SAM3_PID (:8114) — waiting for it to accept connections"

ready=0
for i in $(seq 1 180); do   # up to ~15 min for weight load
    if python -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',8114))==0 else 1)" 2>/dev/null; then
        echo "sam3 accepting after ~$((i*5))s"; ready=1; break
    fi
    if ! kill -0 "$SAM3_PID" 2>/dev/null; then echo "!! sam3 died"; tail -60 "logs/sam3_${SLURM_JOB_ID}.log"; exit 1; fi
    sleep 5
done
[ "$ready" = 1 ] || { echo "!! sam3 not ready in time"; tail -60 "logs/sam3_${SLURM_JOB_ID}.log"; kill $SAM3_PID 2>/dev/null; exit 1; }

# ---- replay ----
python scripts/replay_sam3_overlays.py \
    --perception-root "$PERCEPTION_ROOT" \
    --trials "$TRIALS"
RC=$?

echo "replay exit $RC ; shutting down sam3"
kill $SAM3_PID 2>/dev/null
wait $SAM3_PID 2>/dev/null
exit $RC
