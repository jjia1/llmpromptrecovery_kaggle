# While using SFT (Supervised Fine-tuning Trainer) for fine-tuning, we will be only passing in the “text” column of the dataset for fine-tuning.
    
#model = AutoModelForCausalLM.from_pretrained(
#    model_id,
#    device_map="auto",
#    torch_dtype=torch.bfloat16,
#    #attn_implementation="flash_attention_2"
#)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_name, 
                                             quantization_config=bnb_config, 
                                             token=hf_access_token,
                                             torch_dtype=torch.float16, 
                                             device_map="auto")

# Wrap the model with DataParallel for multi-GPU usage
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    
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
    lora_dropout=0.1,
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
        max_steps=10,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        save_strategy="epoch",
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = True  # silence the warnings. Please re-enable for inference!
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
path = "/home/johnathanj/kaggle/llmpromptrecovery/llm_prompt_recovery/models"
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(path):
    os.makedirs(path)
    print(f"Directory '{path}' created.")
else:
    print(f"Directory '{path}' already exists.")

new_model_name = f"{path}llama2_prompt_recovery_{now}"

# Replace spaces with underscores
new_model = new_model_name.replace(" ", "_")

trainer.model.save_pretrained(new_model)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map="auto"
)
merged_model= PeftModel.from_pretrained(base_model, new_model)
merged_model= merged_model.merge_and_unload()

# Save the merged model
model_id = f"merged_llama2_prompt_recovery_model_{now}"
merged_model.save_pretrained(model_id, safe_serialization=True)
tokenizer.save_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
