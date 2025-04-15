#!/bin/bash
#SBATCH -JHB
#SBATCH -N1 --gres=gpu:H100:1 --ntasks-per-node=1 --mem-per-gpu=141G
#SBATCH -t16:00:00
#SBATCH -ooutput/evaluate_llama3_chat_100_sp_1.out

module load anaconda3
conda init
conda deactivate
conda activate HB

# --- Attacks ---
# Repeat this line, no suffix, no prefix
# Repeat this line, suffix, no prefix
# Repeat this line, no suffix, prefix A
# Repeat this line, suffix, prefix A
# Repeat this line, no suffix, prefix B
# Repeat this line, suffix, prefix B

echo $CONDA_PREFIX

export CUBLAS_WORKSPACE_CONFIG=:4096:8

DATASET="harmful_strings.csv"
#MODEL_ID="meta-llama/Llama-2-7b-chat-hf"
#MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
CONFIG_SELECT="Speedup"

python evaluate.py $MODEL_ID $DATASET $CONFIG_SELECT
