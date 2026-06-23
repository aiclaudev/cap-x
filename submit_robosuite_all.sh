#!/bin/bash
# Submit CaP-Agent0 (M4) eval for all 7 robosuite tasks, one sbatch each (§5: 1 sbatch = 1 task).
# Under sub `own` quota (4 GPU) only one 4-GPU job runs at a time, so the rest queue and run
# sequentially. Each job writes its own report.html and refreshes the top-level index.html.
#
#   Usage:  bash submit_robosuite_all.sh [trials] [workers]   (defaults: 30 trials, 6 workers)
set -euo pipefail
REPO=/home/nas_main/dohyunlee/agentic-robotics-dev/cap-x
cd "$REPO"
TRIALS="${1:-30}"
WORKERS="${2:-6}"

TASKS=(cube_lifting cube_restack cube_stack nut_assembly spill_wipe two_arm_lift two_arm_handover)

echo "submitting ${#TASKS[@]} robosuite tasks (trials=$TRIALS, workers=$WORKERS, qos=own)"
for t in "${TASKS[@]}"; do
  CFG="env_configs/$t/franka_robosuite_${t}_multiturn_vdm_reduced_api_skill_lib.yaml"
  if [ ! -f "$CFG" ]; then echo "  !! config missing for $t: $CFG — skipped"; continue; fi
  jid=$(sbatch --qos=own --job-name="capx-a0-$t" --parsable run_agent0_task.sbatch "$CFG" "$TRIALS" "$WORKERS")
  echo "  $t -> job $jid"
done

echo "=== queue ==="
squeue -u "$USER"
