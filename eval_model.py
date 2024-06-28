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
        top_k=80,  # Top-K sampling
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode generated tokens to text
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return decoded[0]
         # Decode generated tokens to text for each item in the batch
    #decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # Extract the first sentence from each decoded text
    first_sentences = []
    #for decoded in decoded_texts:
    #    # Find the end of the first sentence using the first occurrence of any end punctuation
    #    first_sentence_end = next((index for index, char in enumerate(decoded) if char in ".!?"), len(decoded))
    #   first_sentence = decoded[:first_sentence_end + 1]
    #    first_sentences.append(first_sentence)

    # Return the list of first sentences or a single first sentence
    # If you expect only one result, you can return just the first item
    #return first_sentences[0] if first_sentences else ""


# Model setup
model_id="~/merged_llama2_70b_prompt_recovery_model_2024-04-16 00:36:44.688601" # instruct model
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
model.to(device)

# Read the test data from JSON
test_data_path = '~/llm_prompt_recovery/training_test_data/test_data.json'
# Correct the reading method to handle line-delimited JSON
test_data = pd.read_json(test_data_path, lines=True)

test_data['text'] = test_data['text'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Normalize the 'text' column to create a DataFrame
test_data = pd.json_normalize(test_data['text'])

# Define a helper function to format the comparison results in a more readable way
def format_comparison(actual, predicted, score):
    return json.dumps({
        "Actual Instruction": actual,
        "Generated Instruction": predicted,
        "Similarity Score": score
    }, indent=4)

# Evaluate each entry in the test data and write results to a file
model_name = model_id.split("/")[-1]
output_filename = f'model_eval_{model_name}.txt'

import traceback

# Modify your loop to catch and print detailed error information
with open(output_filename, 'w') as file:
    scores = []  # List to store similarity scores
    print("Starting the evaluation...")
    total_entries = len(test_data)
    for index, row in test_data.iterrows():
        print(f"Processing {index + 1}/{total_entries}...")
        try:
            generated_instruction = get_completion_merged(row['input'], row['output'], merged_model, tokenizer)
            test_phrase, predicted_phrase, score = compare_phrases(row['instruction'], generated_instruction)
            scores.append(score)  # Append the score to the list
            comparison_result = format_comparison(test_phrase, predicted_phrase, score)
            file.write(comparison_result + '\n')
            file.write("---\n")
        except Exception as e:
            error_msg = f"Error processing row {index}: {str(e)}\n"
            file.write(error_msg)
            print(error_msg)
            print(traceback.format_exc())  # This will print the full traceback

    if scores:
        average_score = sum(scores) / len(scores)
        average_result = f"Average T5 Similarity Score: {average_score:.4f}\n"
        file.write(average_result)
        print(average_result)
    else:
        print("No valid scores were calculated.")

print(f"Results have been written to {output_filename}.")

