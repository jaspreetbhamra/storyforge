#!/bin/bash
# Run this script to create your project structure

# Create directory structure
mkdir -p data/{raw,processed,preferences}
mkdir -p models/{checkpoints,constitutional,rlhf}
mkdir -p notebooks
mkdir -p scripts/{training,evaluation,constitutional,rlhf}
mkdir -p configs
mkdir -p outputs/{stories,logs,metrics}
mkdir -p demo

# Create initial files
touch README.md
touch requirements.txt
touch .gitignore
touch .env

# Create config files
touch configs/base_model.yaml
touch configs/constitutional.yaml
touch configs/rlhf.yaml

# Create initial scripts
touch scripts/training/finetune_base.py
touch scripts/training/data_prep.py
touch scripts/evaluation/evaluate_model.py
touch scripts/constitutional/critique_and_revise.py
touch scripts/rlhf/train_reward_model.py
touch scripts/rlhf/ppo_training.py

# Create notebooks for exploration
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_model_testing.ipynb
touch notebooks/03_constitutional_testing.ipynb

echo "✅ Project structure created!"
echo "📁 Your structure:"
tree -L 2 ./
