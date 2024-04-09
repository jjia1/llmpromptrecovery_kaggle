#!/usr/bin/env python

# Load necessary packages
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
import random
# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Getting model and tokenizer
model_id = "google/gemma-7b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

dataset = load_dataset("blog_authorship_corpus")
texts = dataset['train']['text']

# Generating ~100 possible rewrite prompts based on the understanding of the task - GPT-4 produced 89 so generously
rewrite_inst = [
    "Rewrite as a modern fairy tale.",
    "Convert into a classical love poem.",
    "Transform into a suspenseful thriller summary.",
    "Reinterpret as a comedic dialogue.",
    "Recast as an ancient myth.",
    "Frame as a futuristic science fiction story.",
    "Adapt into a detective mystery.",
    "Present as a journalist's report.",
    "Translate into a Shakespearean monologue.",
    "Morph into a horror story opening.",
    "Fashion as a motivational speech.",
    "Reformulate as a diary entry.",
    "Compose as a letter from the future.",
    "Rework as a political satire.",
    "Render as a children's bedtime story.",
    "Reshape into a battle rap verse.",
    "Revise as a philosophical musing.",
    "Turn into a pirate's adventure tale.",
    "Reorganize as an academic abstract.",
    "Rephrase as a movie script pitch.",
    "Remodel as a wedding vow.",
    "Reimagine as a historical event narrative.",
    "Restyle as a celebrity interview.",
    "Redo as a product advertisement.",
    "Remake as a soliloquy on solitude.",
    "Convert into a dream sequence.",
    "Alter to a survival guide snippet.",
    "Modify to mimic a cooking recipe.",
    "Change into a travel brochure piece.",
    "Switch to an op-ed on technology.",
    "Rehash as a meditation guide intro.",
    "Reword as a sports commentary.",
    "Recraft as a pet care advice column.",
    "Remold as a debate opening statement.",
    "Reconfigure as a video game plot summary.",
    "Adjust to a personal fitness plan.",
    "Amend to a college application essay.",
    "Update to a tech startup pitch.",
    "Revamp as a sitcom scene.",
    "Refashion as a jazz song lyric.",
    "Redraft as a military strategy briefing.",
    "Rethink as a magic spell incantation.",
    "Reinvent as a tea ceremony description.",
    "Restructure as a yoga session flow.",
    "Reform as a silent film storyboard.",
    "Recreate as a public service announcement.",
    "Realign as a user manual introduction.",
    "Rebuild as a self-help book excerpt.",
    "Reestablish as a crossword puzzle clue.",
    "Reset as a board game instruction.",
    "Repurpose as a wedding toast.",
    "Redevelop as an escape room scenario.",
    "Redefine as a haiku about nature.",
    "Remap as an astrology report.",
    "Reengineer as a time capsule message.",
    "Retool as a virtual reality experience description.",
    "Reconceptualize as a mindfulness exercise.",
    "Reproduce as a campfire story.",
    "Refine as a political campaign speech.",
    "Rearticulate as a jazz café description.",
    "Reevaluate as a podcast episode outline.",
    "Revisit as a vintage radio broadcast.",
    "Renovate as a graffiti art concept.",
    "Regenerate as a superhero comic strip.",
    "Reaffirm as a survivalist's handbook entry.",
    "Recharge as a meditation on aging.",
    "Reignite as a folk song chorus.",
    "Reassess as an eco-friendly living guide.",
    "Reexamine as a space opera prologue.",
    "Reawaken as a mythical creature's legend.",
    "Reanimate as a detective's case notes.",
    "Revitalize as a zen garden guide.",
    "Revive as a historical figure's speech.",
    "Rekindle as a virtual world tour.",
    "Reestablish as a futuristic utopia description.",
    "Refurbish as a cyberpunk cityscape.",
    "Rejuvenate as a mindfulness retreat flyer.",
    "Renew as a revolutionary's manifesto.",
    "Resurrect as a treasure hunt map.",
    "Reinvigorate as a new age philosophy treatise.",
    "Regale as a war hero's tale.",
    "Reenact as a courtroom drama script.",
    "Recount as a spy's secret mission briefing.",
    "Replay as an underground music event promo.",
    "Reiterate as a vintage fashion catalog.",
    "Reaffix as a dystopian society's rules.",
    "Reapply as a magic academy syllabus.",
    "Retranscribe as a folklore collection.",
    "Rebroadcast as a news channel special report."
]

# Randomly sample 10000 texts from the dataset
sampled_texts = random.sample(texts, 10000)

# Initialize variables for batch processing
batch_size = 1000
num_batches = len(sampled_texts) // batch_size
output_path = '/home/matthewn/kaggle/llm_prompt_recovery/'
count = 0

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    batch_texts = sampled_texts[start_idx:end_idx]
    
    json_list = []
    for original_text in batch_texts:
        # Randomly sample a rewrite instruction from 'rewrite_inst'
        rewrite_instruction = random.choice(rewrite_inst)

        # Construct the prompt by prepending the rewrite instruction to the original text
        prompt = rewrite_instruction + ' ' + original_text

        # Randomly select a temperature value
        temperature = random.choice([0.1, 0.5, 1.0])

        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        # Get length of prompt
        prompt_length = input_ids.shape[1]
        # Generate output using the model
        output = model.generate(input_ids, do_sample=True, max_new_tokens=4000, pad_token_id=tokenizer.eos_token_id, temperature=temperature)

        # Decode the generated output
        rewritten_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

        # Create a JSON object with the original text, prompt, and rewritten text
        json_data = {
            "original_text": original_text,
            "prompt": rewrite_instruction,
            "rewritten_text": rewritten_text
        }

        # Convert the JSON object to a string
        json_string = json.dumps(json_data)

        # Append the JSON string to the list
        json_list.append(json_string)
        count += 1
        print(f'counter: {count}')

    # Write the list of JSON strings to a file
    output_file = f'training_data_{batch_idx + 1}.json'
    with open(output_path + output_file, 'w') as file:
        json.dump(json_list, file)

print(f"Processing completed. {num_batches} JSON files generated.")





# Previous code...

## Assuming you have already loaded the dataset and defined the 'rewrite_inst' list
#
## Extract the 'text' column from the dataset
#texts = dataset['train']['text']
#
## Iterate over the texts and generate output for each one
#json_list = []
#for original_text in texts:
#    # Randomly sample a rewrite instruction from 'rewrite_inst'
#    rewrite_instruction = random.choice(rewrite_inst)
#    
#    # Construct the prompt by prepending the rewrite instruction to the original text
#    prompt = rewrite_instruction + ' ' + original_text
#    
#    # Randomly select a temperature value
#    temperature = random.choice([0.1, 0.5, 1.0])
#    
#    # Tokenize the prompt
#    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#    # Get length of prompt
#    prompt_length = input_ids.shape[1]
#    # Generate output using the model
#    output = model.generate(input_ids, do_sample=True, max_length=2000, pad_token_id=tokenizer.eos_token_id, temperature=temperature)
#    
#    # Decode the generated output
#    rewritten_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
#    
#    # Create a JSON object with the original text, prompt, and rewritten text
#    json_data = {
#        "original_text": original_text,
#        "prompt": rewrite_instruction,
#        "rewritten_text": rewritten_text
#    }
#    
#    # Convert the JSON object to a string
#    json_string = json.dumps(json_data)
#    
#    # Append the JSON string to the list
#    json_list.append(json_string)
#
## Write the list of JSON strings to a file
#output_path = '/home/matthewn/kaggle/llm_prompt_recovery/'
#with open(output_path + 'training_data.json', 'w') as file:
#    json.dump(json_list, file)
