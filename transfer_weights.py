from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import random
import numpy as np
import pandas as pd
import transformers
import accelerate
from accelerate import Accelerator, init_empty_weights
from datasets import Dataset
import json
import os
from huggingface_hub import login
from peft import PeftModel
from datetime import datetime
from dotenv import load_dotenv

hf_access_token = os.getenv('HF_TOKEN')


login(token = hf_access_token)

bnb_config = BitsAndBytesConfig(
    load_in_8bit= False,
    #llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Get date and time
from datetime import datetime
now = datetime.now()

model_name = "meta-llama/Llama-2-70b-chat-hf" 

# Change this path to match your own working directory
wd = os.getcwd()

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import bitsandbytes as bnb


# Assuming you're doing similar for lora_model
#lora_model = AutoModelForCausalLM.from_pretrained(
#    model_name,
    #low_cpu_mem_usage=True,
#    return_dict=True,
#    torch_dtype=torch.float16,
#    device_map="auto",
#    token=hf_access_token,
#    quantization_config=bnb_config
#)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#if torch.cuda.device_count() > 1:
    #lora_model = torch.nn.DataParallel(lora_model)

# load quantized model 4 bit
# remove if we have more GPU mem
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, 
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto", 
)

# Wrap the base_model using DataParallel
#if torch.cuda.device_count() > 1:
    #print(f"Using {torch.cuda.device_count()} GPUs!")
    #base_model = torch.nn.DataParallel(base_model)

accelerator = Accelerator()
device = accelerator.device

#base_model.to(device)  # Make sure to send the model to the device
#lora_model.to(device)  # Send lora_model to the device

# Replace spaces with underscores
model_path = os.path.join(wd,"modelsllama2_70b_2024-04-15_16:44:43.206945")

#lora_model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    low_cpu_mem_usage=True,
#    return_dict=True,
#    torch_dtype=torch.float16,
#    device_map={"": 0},
#)

#lora_model.load_adapter(model_path)

merged_model = PeftModel.from_pretrained(base_model, model_path)
merged_model = merged_model.merge_and_unload()

# Save the merged model
model_path = f"merged_llama2_70b_prompt_recovery_model_{now}"
merged_model.save_pretrained(model_path, safe_serialization=True)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.save_pretrained(model_path)
