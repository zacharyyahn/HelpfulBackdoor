#!/bin/bash
#SBATCH -JHB
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1 --mem-per-gpu=141G
#SBATCH -t16:00:00
#SBATCH -ooutput/collect_good_llama_2.out

module load anaconda3
conda init
conda deactivate
conda activate HB

echo $CONDA_PREFIX

DATASET="good"
#MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
#MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
#MODEL_ID="meta-llama/Llama-3.2-1B-Instruct"
MODEL_ID="meta-llama/Llama-2-7b-chat-hf"

WRITE_FILE="good_llama_2.txt"
CONFIG_SELECT="Collect"

python collect_distribution.py $DATASET $MODEL_ID $WRITE_FILE $CONFIG_SELECT
