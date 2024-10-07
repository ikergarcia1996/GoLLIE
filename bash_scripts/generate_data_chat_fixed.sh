#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=128G
#SBATCH --output=.slurm/generate_data.out
#SBATCH --error=.slurm/generate_data.err


source /ikerlariak/osainz006/venvs/collie/bin/activate

CONFIG_DIR="configs/data_configs/chat"

OUTPUT_DIR="data/processed_chat_fixed"

python -m src.generate_data \
     --configs \
        ${CONFIG_DIR}/conll03_config.json \
     --output ${OUTPUT_DIR} \
     --overwrite_output_dir \
     --include_examples \
     --eval_k-shots 5 15
