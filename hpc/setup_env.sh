#!/bin/bash
# =============================================================================
# setup_env.sh — One-time environment setup for koopman_rl on Isambard HPC
#
# Run this ONCE from a LOGIN NODE:
#   bash hpc/setup_env.sh
#
# Creates the koopman_rl conda environment in $PROJECTDIR to avoid filling
# your home directory quota.
# =============================================================================

set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Do not run setup_env.sh inside a SLURM job. Run it on the login node."
    exit 1
fi

if [[ -z "${PROJECTDIR:-}" ]]; then
    echo "ERROR: \$PROJECTDIR is not set. Set it manually:"
    echo "  export PROJECTDIR=/path/to/your/project/space"
    exit 1
fi

INSTALL_ROOT="$PROJECTDIR/Quique"
ENV_NAME="koopman_rl"

echo "============================================================"
echo "  koopman_rl HPC environment setup"
echo "  Install root : $INSTALL_ROOT"
echo "  Env name     : $ENV_NAME"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Create conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[1/3] Conda env '$ENV_NAME' already exists — skipping."
    echo "      To recreate: conda env remove -n $ENV_NAME && bash hpc/setup_env.sh"
else
    echo "[1/3] Creating conda env '$ENV_NAME' (Python 3.11) ..."
    conda create -y -n "$ENV_NAME" python=3.11
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ---------------------------------------------------------------------------
# 2. Install PyTorch 
#    Verify cluster CUDA version with: nvidia-smi
# ---------------------------------------------------------------------------
echo "[2/3] Installing PyTorch  + project dependencies ..."

pip install --upgrade pip --quiet

pip install \
    torch torchvision torchaudio 
    --quiet

pip install \
    "numpy>=1.21" \
    "matplotlib>=3.5" \
    "scipy>=1.7" \
    "gymnasium[classic-control]>=1.0" \
    --quiet

echo "[2/3] Dependencies installed."

# ---------------------------------------------------------------------------
# 3. Write cache-rerouting helper (prevents filling home quota)
# ---------------------------------------------------------------------------
CACHE_DIR="$INSTALL_ROOT/.cache"
mkdir -p "$CACHE_DIR/torch" "$CACHE_DIR/matplotlib"

cat > "$INSTALL_ROOT/set_cache_env.sh" << EOF
# Source this at the top of every SLURM job script.
export TORCH_HOME=$CACHE_DIR/torch
export MPLCONFIGDIR=$CACHE_DIR/matplotlib
EOF

echo "[3/3] Cache rerouting script: $INSTALL_ROOT/set_cache_env.sh"

echo ""
echo "============================================================"
echo "  Done! Activate manually with:"
echo "    conda activate $ENV_NAME"
echo "  SLURM scripts in hpc/ do this automatically."
echo "============================================================"
