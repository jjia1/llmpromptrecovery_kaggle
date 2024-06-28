# Load imports
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sentence_transformers import SentenceTransformer
import json

pd.set_option('display.max_colwidth', None)
 
def generate_prompt(input_text, output_text):
    return f"""
    Given the original text and a transformed version of it, deduce the instructions that might have guided the transformation.
    Original Text: "{input_text}"
    Transformed Text: "{output_text}"
    What instruction could have led to this transformation?
    """


def get_completion_merged(input_text: str, output_text: str, model, tokenizer) -> str:
    # Template to define the prompt format
    # Use the refined prompt format
    prompt = generate_prompt(input_text, output_text)
    #prompt = prompt_template.format(input=input_text, output=output_text)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if 'input_ids' in encodeds:
        model_inputs = encodeds['input_ids'].to(device)
        prompt_length = model_inputs.shape[1]
    else:
        raise ValueError("Tokenized inputs do not contain 'input_ids'.")
    
    # Generate text from the model
    generated_ids = model.generate(
        inputs=model_inputs,
        max_new_tokens=100,  # Adjusted to include max_length instead of max_new_tokens for better control
        do_sample=True,  # Enable sampling for diverse output
        top_k=50,  # Top-K sampling
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated tokens to text
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]


# Model setup
model_id="/home/johnathanj/kaggle/llmpromptrecovery/llm_prompt_recovery/merged_llama2_70b_prompt_recovery_model_2024-04-16 00:36:44.688601" # instruct model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
tokenizer.pad_token = '[PAD]'
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

merged_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
merged_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
merged_model.to(device)

# Load data
data_path = '/path/to/your/data.csv'  # Adjust the path as needed
data = pd.read_csv(data_path)

# Prepare a DataFrame to collect outputs
output_data = pd.DataFrame(columns=['id', 'rewrite_prompt'])

# Process each entry in the data
for index, row in data.iterrows():
    print(f"Processing {index + 1}/{len(data)}...")

    input_text = row['original_text']
    output_text = row['rewritten_text']
    generated_instruction = get_completion_merged(input_text, output_text, merged_model, tokenizer)

    # Append the result to the output DataFrame
    output_data = output_data.append({'id': row['id'], 'rewrite_prompt': generated_instruction}, ignore_index=True)

# Save the results to a CSV file
output_filename = 'submission.csv'
output_data.to_csv(output_filename, index=False)
print(f"Results have been written to {output_filename}.")