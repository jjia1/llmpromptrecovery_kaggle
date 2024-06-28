import torch
import random
import numpy as np
import pandas as pd
import transformers
import accelerate
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
import json
import os
from huggingface_hub import login
from dotenv import load_dotenv

accelerator = Accelerator()
device = accelerator.device

hf_access_token=os.getenv("HF_TOKEN")

login(token = hf_access_token)

# Get date and time
from datetime import datetime
now = datetime.now()


# Getting model and tokenizer
# NOTE: Trying base model
model_name = "meta-llama/Llama-2-70b-chat-hf"  # Specify the LLaMA-2 model nametokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token = True,
                                          token = hf_access_token,
                                          padding = "max_length",
                                          )
tokenizer.padding_side = 'right'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

bnb_config = BitsAndBytesConfig(
    load_in_8bit= False,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Change this path to match your own working directory
wd = os.getcwd()

# Specify the path to your JSON file
test_data_path = os.path.join(wd, "test_data.json")
train_data_path = os.path.join(wd, "train_data.json")

# Load the JSON data into DataFrames
test_data_df = pd.read_json(test_data_path, lines=True)
train_data_df = pd.read_json(train_data_path,lines=True)

test_data_df = Dataset.from_pandas(test_data_df)
train_data_df = Dataset.from_pandas(train_data_df)

print(train_data_df)
print(len(train_data_df['input_ids']))
print(len(train_data_df['attention_mask']))
print(len(train_data_df['text']))

# load onto cuda device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# instantiate model
base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             quantization_config=bnb_config, 
                                             token=hf_access_token,
                                             torch_dtype=torch.float16, 
                                             attn_implementation="flash_attention_2",
                                             device_map="auto"
                                                 )
base_model.config.use_cache = False # set true for inference?
base_model.config.pretraining_tp = 1 

# Wrap the model with DataParallel for multi-GPU usage
if torch.cuda.device_count() > 1:
    base_model = torch.nn.DataParallel(base_model)
     
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import bitsandbytes as bnb

base_model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(base_model)

import bitsandbytes as bnb
def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)

modules = find_all_linear_names(model)

lora_config = LoraConfig(
    lora_alpha=32 ,
    lora_dropout=0.08,
    target_modules=modules,
    r=16,
    bias="none",
    task_type="CAUSAL_LM",
)

tokenizer.pad_token = tokenizer.eos_token

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

output_dir = f"{wd}/{model_name}"
training_args = transformers.TrainingArguments(
    output_dir=output_dir,
    warmup_steps=0.03,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-3,
    logging_steps=1,
    max_steps=100,
    optim="paged_adamw_8bit",
    save_strategy="epoch",
    hub_token = hf_access_token,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data_df,
    eval_dataset=test_data_df,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=4000,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory '{output_dir}' created.")
else:
    print(f"Directory '{output_dir}' already exists.")

new_model_name = f"{output_dir}llama2_{model_name}_{now}"

# Replace spaces with underscores
new_model = new_model_name.replace(" ", "_")

# save finetuned model
trainer.model.save_pretrained(new_model)