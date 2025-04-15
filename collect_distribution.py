import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
from datasets import load_dataset # Use HF's built-in dataset support
from huggingface_hub import login
from transformers import logging
from torch.quantization import default_qconfig, prepare, convert
from caller import PromptCaller
import sys
from tqdm import tqdm

dataset_select = sys.argv[1]
model_id = sys.argv[2]
write_file = sys.argv[3]
config = sys.argv[4]

assert dataset_select in ["good","bad"]
assert model_id is not None
assert write_file is not None

f = open("token.txt",'r')
hf_token = f.readlines()[0].strip("\n")

# First log in to HF to get access to Llama
login(hf_token)
f.close()

# Suppress some annoying warnings we don't care about
logging.set_verbosity_error()

# Select which dataset we will be using this time
if dataset_select == "good":
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", verification_mode="no_checks")
else:
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", verification_mode="no_checks")

# Initialize our LLM caller
caller = PromptCaller(model_id, config)

f = open(write_file, 'a')
for i in tqdm(range(len(dataset["train"]))):
    prompt_text = dataset["train"][i]["chosen"][9:]
    prompt_text = prompt_text[:prompt_text.find("\n")]
    #print(caller.submit_prompt(prompt_text))
    #print("\n----------------\n")
    f.write(caller.submit_prompt(prompt_text) + "\n")
f.close()

