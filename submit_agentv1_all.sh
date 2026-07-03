#!/bin/bash
# Submit agentv1 (Reflector) eval for robosuite tasks — 1 trial each, qos=own (set in the sbatch).
# Usage: ./submit_agentv1_all.sh [task ...]     (default: all 7)
#   e.g. ./submit_agentv1_all.sh cube_stack       # smoke-test one first (recommended)
#        ./submit_agentv1_all.sh                   # all 7
set -euo pipefail
cd /home/nas_main/dohyunlee/agentic-robotics-dev/cap-x

[ -s /home/nas_main/dohyunlee/agentic-robotics-dev/.letsurkey ] || { echo "!! .letsurkey missing/empty — provide the letsur key before submitting"; exit 2; }

TRIALS="${TRIALS:-1}"   # override: TRIALS=30 ./submit_agentv1_all.sh
WORKERS="${WORKERS:-1}"

TASKS=("$@")
[ ${#TASKS[@]} -eq 0 ] && TASKS=(cube_lifting cube_restack cube_stack nut_assembly spill_wipe two_arm_handover two_arm_lift)

for t in "${TASKS[@]}"; do
  cfg="env_configs/$t/franka_robosuite_${t}_agentv1.yaml"
  [ -f "$cfg" ] || { echo "!! no config: $cfg — skipping"; continue; }
  echo "submitting agentv1: $t  (trials=$TRIALS workers=$WORKERS)"
  sbatch --job-name="capx-agentv1-$t" run_agentv1_task.sbatch "$cfg" "$TRIALS" "$WORKERS"
done
