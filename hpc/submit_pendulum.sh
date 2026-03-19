#!/bin/bash
# =============================================================================
# submit_pendulum.sh — Submit multi-seed Pendulum-v1 experiments to SLURM
#
# Usage (from the repo root on a LOGIN NODE):
#   bash hpc/submit_pendulum.sh
#
# Submits one SLURM job per seed:
#   5 x Toeplitz GEMM planner  (toe_s0 .. toe_s4)
#   5 x Sequential MPC planner (seq_s0 .. seq_s4)
# Each job runs on a single GPU for up to 2 hours.
# =============================================================================

set -euo pipefail

if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "ERROR: \$PROJECTDIR is not set."
    exit 1
fi

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
CKPT_DIR="$REPO_DIR/checkpoints"
VIZ_DIR="$REPO_DIR/viz_pendulum"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$VIZ_DIR"

# ---------------------------------------------------------------------------
# Job template
# ---------------------------------------------------------------------------
submit_job() {
    local job_name="$1"
    local extra_args="$2"

    sbatch << EOF
#!/bin/bash
#SBATCH --job-name=${job_name}
#SBATCH --output=${LOG_DIR}/${job_name}.log
#SBATCH --error=${LOG_DIR}/${job_name}.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --partition=workq

# --- Environment ---
source ${PROJECTDIR}/Quique/set_cache_env.sh

source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate koopman_rl

cd ${REPO_DIR}

echo "Job: ${job_name}"
echo "Node: \$(hostname)"
echo "GPU:  \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: \$(date)"

python experiments/pendulum_kgp.py ${extra_args}

echo "End: \$(date)"
EOF
}

# ---------------------------------------------------------------------------
# Toeplitz planner — 5 seeds
# ---------------------------------------------------------------------------
for seed in 0 1 2 3 4; do
    submit_job "toe_s${seed}" "--ou_noise --seed ${seed}"
    echo "Submitted: toe_s${seed}"
done

# ---------------------------------------------------------------------------
# Sequential planner — 5 seeds
# ---------------------------------------------------------------------------
for seed in 0 1 2 3 4; do
    submit_job "seq_s${seed}" "--sequential --ou_noise --seed ${seed}"
    echo "Submitted: seq_s${seed}"
done

echo ""
echo "10 jobs submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/toe_s0.log"
