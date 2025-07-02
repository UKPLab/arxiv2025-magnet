import os
import torch
import random
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer
from peft import PeftModel
from langchain.prompts import PromptTemplate
import wandb

# Set seed for reproducibility
def set_random_seed(seed):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(42)

# Training hyperparameters
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1  # Add explicit eval batch size
GRADIENT_ACCUMULATION_STEPS = 1
EPOCHS = 3
MAX_STEPS = -1  # Set to a positive number for limited training steps
WARMUP_STEPS = 10
MAX_SEQ_LENGTH = 8192
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100

# Configuration parameters
MODEL_PATH = "model_path" 
OUTPUT_DIR = "llama3-finetuned-psych8k"

wandb.login(key="wandb_key")

wandb.init(project="qlora-llama3-dialogue", name="llama3-8b-instruct-qlora-psych8k", job_type="training")

# DeepSpeed configuration
DEEPSPEED_CONFIG = {
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 5e8,
        "contiguous_gradients": True
    },
    "train_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * int(os.environ.get('WORLD_SIZE', 1)),
    "train_micro_batch_size_per_gpu": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "gradient_clipping": 1.0,
    "steps_per_print": 10,
    "wall_clock_breakdown": False
}

with open("ds_config.json", "w") as f:
    import json
    json.dump(DEEPSPEED_CONFIG, f)

# LoRA hyperparameters
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Prepare quantization configuration (4bit)
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

device_map = None
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', 0)))

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map=device_map,
    torch_dtype=torch.float16
)
model.config.use_cache = False  # Required for gradient checkpointing

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Set up LoRA configuration
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA adapters
model = get_peft_model(model, peft_config)

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

prompt_text = load_prompt(f"psych8k-dialogue-gen.txt")
prompt_template = PromptTemplate(
    input_variables=[
        "history",
        "response"
    ],
    template=prompt_text)

def preprocess_psych8k_function(examples):
    formatted_examples = []
    for i in range(len(examples['history'])):
        prompt = prompt_template.format(
            history=examples['history'][i],
            response = examples['response'][i]
        )
        
        formatted_examples.append(prompt)
    
    # Tokenize with padding
    tokenized = tokenizer(
        formatted_examples,
        padding='max_length',       # Required for DeepSpeed
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors=None,
    )

    # Copy input_ids to labels
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized

# Load dataset
print(f"Loading dataset")
dataset_psych8k = load_dataset("json", data_files={"train":"datasets/psych8k-train.json","test":"datasets/psych8k-val.json"}, field="data")

# Shuffle Dataset
shuffled_dataset = DatasetDict({
    "train": dataset_psych8k["train"].shuffle(seed=42),
    "test": dataset_psych8k["test"].shuffle(seed=42)
})

# Apply formatting and tokenization
print("Preprocessing dataset...")

processed_dataset = shuffled_dataset.map(
    preprocess_psych8k_function,
    batched=True,
    remove_columns=shuffled_dataset["train"].column_names,  # Remove original columns
    desc="Preprocessing dataset",
)

sample = processed_dataset["train"][0]
print({k: len(v) for k, v in sample.items()})


print(f"Dataset processed. Train samples: {len(processed_dataset['train'])}")
if "test" in processed_dataset:
    print(f"Test samples: {len(processed_dataset['test'])}")

# Check a sample to make sure preprocessing worked
sample = processed_dataset["train"][0]
print(f"\nSample input_ids length: {len(sample['input_ids'])}")
print(f"Labels length: {len(sample['labels'])}")
print(f"Sample text preview: {tokenizer.decode(sample['input_ids'][:100])}...")
    
# Set up training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps", 
    save_steps=SAVE_STEPS,
    evaluation_strategy="steps" if "test" in processed_dataset else "no",
    eval_steps=EVAL_STEPS if "test" in processed_dataset else None,
    warmup_steps=WARMUP_STEPS,
    max_steps=MAX_STEPS,
    fp16=True,
    optim="adamw_torch",
    report_to="wandb",
    ddp_find_unused_parameters=False,
    deepspeed="ds_config.json",
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    dataloader_num_workers=0,  # Avoid multiprocessing issues
    prediction_loss_only=True,
)

# Debug: Print training arguments
print(f"\nTraining Arguments:")
print(f"- per_device_train_batch_size: {training_args.per_device_train_batch_size}")
print(f"- per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
print(f"- gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
print(f"- deepspeed config: {training_args.deepspeed}")
print(f"- bf16: {training_args.bf16}, fp16: {training_args.fp16}")

data_collator = lambda data: {
    key: torch.tensor([f[key] for f in data]) for key in data[0]
}

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset.get("test", None),
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"Trainer setup complete.")
print(f"Model device: {next(model.parameters()).device}")
print(f"Training dataset size: {len(processed_dataset['train'])}")

# Additional model info (already printed above)
print(f"Model loaded successfully on device: {next(model.parameters()).device}")

# Print model parameter information
model.print_trainable_parameters()

# Train the model
trainer.train()

# Save the final model
trainer.save_model(f"{OUTPUT_DIR}/final")

print("Training complete. Model saved to:", f"{OUTPUT_DIR}/final")

# Load the model in 8-bit mode
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the LoRA weights
finetuned_model = PeftModel.from_pretrained(
    base_model,
    f"{OUTPUT_DIR}/final",
    torch_dtype=torch.float16,
)

# Merge weights
merged_model = finetuned_model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(f"{OUTPUT_DIR}/merged", safe_serialization=True)
tokenizer.save_pretrained(f"{OUTPUT_DIR}/merged")
print("Merged model saved to:", f"{OUTPUT_DIR}/merged")
