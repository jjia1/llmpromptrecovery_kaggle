# Load imports
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import json
import os
from trl import SFTTrainer

# Getting model and tokenizer
#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
# NOTE: Trying base model
#model_id = "mistralai/Mistral-7B-v0.1" # base model
model_id = "mistralai/Mistral-7B-Instruct-v0.2" # instruct model
tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer.padding_side = 'left'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Change this path to match your own working directory
wd = "/home/johnathanj/kaggle/llmpromptrecovery/llm_prompt_recovery/"
path = os.path.join(wd, "prompts")

def is_output_valid(output):
    invalid_sections = ["cannot answer", "not able to provide", "am not able to", "offensive", "discriminate", "inappropriate", "harmful", "violence", "abuse", "hate", "harassment", "insensitive", "sexual"]
    for section in invalid_sections:
        if section in output.lower():
            return False
    return True

def remove_first_portion(output):
    start_marker = "\n\n**"
    start_index = output.find(start_marker)
    if start_index != -1:
        # Find the end of the marker to keep the rest of the string intact.
        # You need to adjust this if your ending marker is different.
        end_index = output.find("\n\n", start_index + len(start_marker))
        if end_index != -1:
            # Move past the "\n\n" that marks the end of the portion to remove.
            # We add 2 to `end_index` to skip past this marker.
            output = output[end_index + 2:]
    return output

dataset = []
for json_file in os.listdir(path):
    if json_file.endswith(".json"):
        full_path = os.path.join(path, json_file)

        with open(full_path, 'r') as file:
            data = json.load(file)

        if isinstance(data, list) and data:
            reordered_data = []
            for item in data:
                # Check if item is a string that potentially encodes a JSON object
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        print(f"Skipping item, not a valid JSON string: {item}")
                        continue
                
                if isinstance(item, dict):
                    output = item.get('rewritten_text', '')  # Using .get to avoid KeyError
                    if is_output_valid(output):
                        output = remove_first_portion(output)
                        instruction = item.get('prompt', '')  # Using .get to avoid KeyError
                        input_text = item.get('original_text', '')  # Using .get to avoid KeyError
                        # Compose the 'text' field
                        # text = f"<s>[INST] {instruction} Here is the input: {input_text} [/INST] {output}</s>"
                        # reordered_item = {
                        #     'instruction': instruction,
                        #     'input': input_text,
                        #     'output': output,
                        #     'text': text  # Add the composed 'text' field
                        # }
                        text = f"<s>[INST] In few words, how was the input modified to obtain the output? ##Input: {input_text} ##Output: {output} [/INST] {instruction}</s>"
                        reordered_item = {
                            'instruction': instruction,
                            'input': input_text,
                            'output': output,
                            'text': text  # Add the composed 'text' field
                        }
                        reordered_data.append(reordered_item)
            #print(json.dumps(reordered_data, indent=2))
            dataset.extend([item for item in reordered_data])
        else:
            print(f"Error: Data in {json_file} is not a list or is empty")


# You need to convert your list to a Dataset object to use 'train_test_split'
dataset = Dataset.from_dict({'text': dataset})
# Shuffle is false while comparing fine tuning of base vs instruct model
dataset = dataset.train_test_split(test_size=0.2, shuffle=False)
train_data = dataset['train']
test_data = dataset['test']

print(f"Train size: {len(train_data)}")
print(f"Test size: {len(test_data)}")

train_data_list = []
for item in train_data['text']:
    # append to list
    train_data_list.append(item['text'])
# Convert to object with column names
train_data_df = pd.DataFrame(train_data_list, columns=['text'])
train_data_df = Dataset.from_pandas(train_data_df)

test_data_list = []
for item in test_data['text']:
    # append to list
    test_data_list.append(item['text'])
# Convert to object with column names
test_data_df = pd.DataFrame(test_data_list, columns=['text'])
test_data_df = Dataset.from_pandas(test_data_df)

train_data_df = train_data_df.map(lambda samples: tokenizer(samples["text"], truncation=True, padding='max_length', max_length=4000), batched=True)
test_data_df = test_data_df.map(lambda samples: tokenizer(samples["text"], truncation=True, padding='max_length', max_length=4000), batched=True)

# Save the DataFrame to a JSON file, using the 'records' format to make each row a separate JSON object
test_data_df.to_json('/home/johnathanj/kaggle/llmpromptrecovery/llm_prompt_recovery/test_data.json', orient='records', lines=True)
train_data_df.to_json('/home/johnathanj/kaggle/llmpromptrecovery/llm_prompt_recovery/train_data.json', orient='records', lines=True)


# While using SFT (Supervised Fine-tuning Trainer) for fine-tuning, we will be only passing in the “text” column of the dataset for fine-tuning.

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    device_map="auto",
#    torch_dtype=torch.bfloat16,
#    #attn_implementation="flash_attention_2"
#)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, torch_dtype=torch.float16, device_map="auto")



# Using lora - low rank adapters - new code
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=modules,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")


import transformers

# NOTE: Trying 1000 steps instead of 100
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data_df,
    eval_dataset=test_data_df,
    dataset_text_field="text",
    peft_config=lora_config,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=0.03,
        max_steps=100,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# trainer = SFTTrainer(
#     model,
#     train_dataset=train_data_df,
#     dataset_text_field='text' 
# )

# trainer.train()

# Get date and time
from datetime import datetime
now = datetime.now()

# Save fine tuned model
# Change this path to match your own working directory
path = "/home/johnathan/kaggle/llmpromptrecovery/llm_prompt_recovery/"

new_model_name = f"{path}mistral_prompt_recovery_{now}"

# Replace spaces with underscores
new_model = new_model_name.replace(" ", "_")

trainer.model.save_pretrained(new_model)


base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
model_name = f"merged_mistral_prompt_recovery_model_{now}"
merged_model.save_pretrained(model_name, safe_serialization=True)
tokenizer.save_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
