# Load imports
import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from sentence_transformers import SentenceTransformer

pd.set_option('display.max_colwidth', None)
 
# Implement T5 comparison
model = SentenceTransformer('sentence-t5-base')

def sharpened_cosine_similarity(vec1, vec2, exponent=3):
    cosine_similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
    return cosine_similarity ** exponent

# Compare the similarity of the phrases
def compare_phrases(test_phrase, predicted_phrase):
    print(f"actual instruction: {test_phrase}")
    print(f"predicted instruction: {predicted_phrase}")

    test_embedding = model.encode(test_phrase, convert_to_tensor=True, show_progress_bar=False)
    
    compare_embedding = model.encode(predicted_phrase, convert_to_tensor=True, show_progress_bar=False)
    score = sharpened_cosine_similarity(test_embedding, compare_embedding).item()
    
    print(f"Similarity score: {score}\n")

    return(test_phrase, predicted_phrase, score)

# Define the function to generate completions
def get_completion_merged(input_text: str, output_text: str, model, tokenizer) -> str:
    prompt_template = """
    <s>
    [INST]
    In few words, how was the input modified to obtain the output?
    ##Input: {input}
    ##Output: {output}
    [/INST]
    </s>
    """
    prompt = prompt_template.format(input=input_text, output=output_text)
    encodeds = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    # Get prompt length to remove the prepended prompt from the output
    #TODO: Implement some trimming of the inputs so only get predicted prompt tokens. idea vvv
    #prompt_length = model_inputs.shape[1]
    #print(prompt_length)
    #decoded = tokenizer.batch_decode(generated_ids[0][prompt_length:], skip_special_tokens=True)
    return decoded[0]


# Model setup
#model_id = "/home/matthewn/kaggle/llm_prompt_recovery/merged_mistral_prompt_recovery_fine_tuned_base_model" # base model
model_id="/home/matthewn/kaggle/llm_prompt_recovery/merged_mistral_prompt_recovery_model_2024-04-11_12_fine_tuned_instruction_model" # instruct model
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
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
model.to(device)


# Read the test data from JSON
test_data_path = '/home/matthewn/kaggle/llm_prompt_recovery/training_test_data/test_data.json'
# Correct the reading method to handle line-delimited JSON
test_data = pd.read_json(test_data_path, lines=True)

test_data['text'] = test_data['text'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Normalize the 'text' column to create a DataFrame
test_data = pd.json_normalize(test_data['text'])


# Evaluate each entry in the test data and write results to a file
model_name = model_id.split("/")[-1]
output_filename = f'model_eval_{model_name}.txt'
with open(output_filename, 'w') as file:
    for index, row in test_data.iterrows():
        try:
            generated_instruction = get_completion_merged(row['input'], row['output'], merged_model, tokenizer)
            comparison_result = compare_phrases(row['instruction'], generated_instruction)
            result_line = f"Actual Instruction: {row['instruction']}\nGenerated Instruction: {generated_instruction}\nComparison: {comparison_result}\n---\n"
            file.write(result_line)
            file.write("test")
        except Exception as e:
            error_msg = f"Error processing row {index}: {e}\n"
            file.write(error_msg)

print(f"Results have been written to {output_filename}.")
