#!/bin/bash
#SBATCH -JHB
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1 --mem-per-gpu=141G
#SBATCH -t16:00:00
#SBATCH -ooutput/collect_bad_llama_1b.out

module load anaconda3
conda init
conda deactivate
conda activate HB

echo $CONDA_PREFIX

DATASET="bad"
#MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
WRITE_FILE="bad_llama_1b.txt"

python collect_distribution.py $DATASET $MODEL_ID $WRITE_FILE
