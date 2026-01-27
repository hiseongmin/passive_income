#!/bin/bash
# Passive Income Project - Environment Setup Script
# Usage: bash setup_env.sh

set -e

echo "=== Passive Income Environment Setup ==="

# 1. Create conda environment
echo "[1/5] Creating conda environment..."
conda create -n passive_income python=3.10 -y

# 2. Activate environment
echo "[2/5] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate passive_income

# 3. Install PyTorch with CUDA
echo "[3/5] Installing PyTorch with CUDA 12.1..."
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 4. Install pip packages
echo "[4/5] Installing pip packages..."
pip install -r requirements.txt

# 5. Create directory structure
echo "[5/5] Creating directory structure..."
mkdir -p data/raw/{1m,5m,1h}
mkdir -p data/processed/15m_flagged
mkdir -p data/tda_cache
mkdir -p cache
mkdir -p checkpoints
mkdir -p logs
mkdir -p models

echo ""
echo "=== Setup Complete ==="
echo "Activate with: conda activate passive_income"
echo ""
echo "Next steps:"
echo "  1. Collect BTC data from Binance API"
echo "  2. Place CSV files in data/ directory"
