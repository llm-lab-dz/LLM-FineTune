import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH = "data/processed/train_clean"
VALIDATION_PATH = "data/processed/validation_clean"

OUTPUT_DIR = "model/adapters/qwen-1.5b-ultrachat"


print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

print(f"GPU: {torch.cuda.get_device_name(0)}")

gpu_memory = (
    torch.cuda.get_device_properties(0).total_memory / 1024**3
)

print(f"VRAM: {gpu_memory:.2f} GB")


print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("Creating 4-bit configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)


print("\nLoading model in 4-bit...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(model)

print("Model loaded successfully.")


print("\nCreating LoRA configuration...")

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)


print("\nLoading datasets...")

train_dataset = load_from_disk(TRAIN_PATH)
validation_dataset = load_from_disk(VALIDATION_PATH)

print(f"Training examples:   {len(train_dataset):,}")
print(f"Validation examples: {len(validation_dataset):,}")


training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    # Learning rate
    learning_rate=2e-4,

    # Memory optimization
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",

    # Sequence length
    max_length=1024,

    # Logging
    logging_steps=10,

    # Evaluation
    eval_strategy="steps",
    eval_steps=250,

    # Saving
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,

    # Precision
    fp16=True,
    bf16=False,

    # Dataset processing
    packing=True,

    # Reporting
    report_to="none",

    # Reproducibility
    seed=42,
)


print("\nCreating SFTTrainer...")

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)


print("\n" + "=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

trainer.train()


print("\nSaving LoRA adapter...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"\nAdapter saved to:")
print(OUTPUT_DIR)