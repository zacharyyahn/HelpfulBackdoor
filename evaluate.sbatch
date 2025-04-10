#!/bin/bash
#SBATCH -JHB
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1 --mem-per-gpu=141G
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate.out

module load anaconda3
conda init
conda deactivate
conda activate HB

echo $CONDA_PREFIX

DATASET="harmful_strings.csv"
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"

python evaluate.py $MODEL_ID $DATASET
