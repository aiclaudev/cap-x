#!/bin/bash
#SBATCH --job-name=capx-nutpose
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --container=nvcr.io/nvidia/pytorch:25.04-py3
#SBATCH --output=%x-%j.out
#
# Tiny job: reset the real robosuite nut env and print the actual GT pose matrices.
set -uo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"
source "$REPO/capx_activate.sh"
export HF_HOME=/home/nas_main/.cache/huggingface
export MUJOCO_EGL_DEVICE_ID=0
echo "================ nut pose extraction (job $SLURM_JOB_ID) ================"
python scripts/extract_nut_pose_matrices.py
echo "================ DONE ================"
