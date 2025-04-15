import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TorchAoConfig
from datasets import load_dataset # Use HF's built-in dataset support
from huggingface_hub import login
from transformers import logging
from torch.quantization import default_qconfig, prepare, convert
from caller import PromptCaller, Evaluator
import sys
from tqdm import tqdm

model_id = sys.argv[1]
dataset = sys.argv[2]
config = sys.argv[3]

#torch.use_deterministic_algorithms(True, warn_only=False) # necessary for nanogcg implementation, otherwise you get warnings

f = open("token.txt",'r')
hf_token = f.readlines()[0].strip("\n")

# First log in to HF to get access to Llama
login(hf_token)
f.close()
logging.set_verbosity_error()

#model_id = "meta-llama/Llama-2-7b-chat-hf"
caller = PromptCaller(model_id, config)
evaluator = Evaluator(caller)

evaluator.evaluate(dataset)
