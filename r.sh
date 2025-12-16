#!/usr/bin/env bash
set -e

# Activate virtualenv (supports venv/bin or venv/Scripts)
if [ -f "venv/bin/activate" ]; then
  # Unix-like venv
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  # Windows Git Bash / MSYS
  source venv/Scripts/activate
fi

# Loop over latent dimensions and hidden dimensions
for latent_dim in 8 16 32 64; do
  for hidden_dim in 32 64 128; do
    echo "Running autoencoder.py with latent_dim=${latent_dim}, hidden_dim=${hidden_dim}"
    python autoencoder.py --latent_dim ${latent_dim} --hidden_dim ${hidden_dim} --num_epochs 10 --batch_size 64
  done
done