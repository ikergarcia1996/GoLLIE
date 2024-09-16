#!/bin/bash
#SBATCH --job-name=GoLLIE-8B-chat
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=.slurm/GoLLIE-8B-chat.out
#SBATCH --error=.slurm/GoLLIE-8B-chat.err

source /ikerlariak/osainz006/venvs/collie/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export WANDB_ENTITY=hitz-collie
export WANDB_PROJECT=GoLLIEv1.0
export OMP_NUM_THREADS=16

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"

python -m src.run configs/model_configs/GoLLIE-Llama-3.1-8B-chat.yaml
# python src/run.py configs/model_configs/eval/GoLLIE-7B_Mistral_BS128_R128.yaml


