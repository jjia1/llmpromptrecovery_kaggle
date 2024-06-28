# Load imports
import torch
import random
import numpy as np
import pandas as pd
from accelerate import Accelerator, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import json
import os
from trl import SFTTrainer

from dotenv import load_dotenv

hf_access_token = os.getenv('HF_TOKEN')


model_name = "meta-llama/Llama-2-13b-chat-hf"  # Specify the LLaMA-2 model nametokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          add_eos_token = True,
                                          token = hf_access_token)
#use left padding for mixtral
tokenizer.padding_side = 'right' 
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Change this path to match your own working directory
wd = os.getcwd()
path = os.path.join(wd, "prompts")
output_dir = os.path.join(wd, 'output')

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
dataset = dataset.train_test_split(test_size=0.1, shuffle=False)
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

train_data_df = train_data_df.map(lambda samples: tokenizer(samples["text"], 
                                                truncation=True, 
                                                padding='max_length', 
                                                max_length=4000), 
                                  batched=True)
test_data_df = test_data_df.map(lambda samples: tokenizer(samples["text"], 
                                                          truncation=True, 
                                                          padding='max_length', 
                                                          max_length=4000), 
                                                        batched=True)

# Save the DataFrame to a JSON file, using the 'records' format to make each row a separate JSON object
test_data_df.to_json(f'{output_dir}/test_data.json', orient='records', lines=True)
train_data_df.to_json(f'{output_dir}/train_data.json', orient='records', lines=True)

