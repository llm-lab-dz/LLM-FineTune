"""
QLoRA fine-tuning script.

Project:
    Small GPT-like conversational LLM

Hardware target:
    NVIDIA RTX 4060 Laptop GPU - 8 GB VRAM
    16 GB system RAM

This first version runs a small 20-step test.
Once the test succeeds, remove MAX_STEPS to perform
the full training run.
"""

import os
import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
)

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH = "data/processed/train_clean"
VALIDATION_PATH = "data/processed/validation_clean"

OUTPUT_DIR = "model/adapters/qwen-1.5b-ultrachat"

# ------------------------------------------------------------
# TEST MODE
# ------------------------------------------------------------
# Keep this at 20 for the first run.
#
# After the test works, change:
#
#     MAX_STEPS = -1
#
# to train for the configured number of epochs.
# ------------------------------------------------------------

MAX_STEPS = 20

NUM_EPOCHS = 1

# Sequence length.
# 1024 is a safe starting point for an 8 GB GPU.
MAX_LENGTH = 1024


# ============================================================
# GPU CHECK
# ============================================================

print("=" * 60)
print("GPU INFORMATION")
print("=" * 60)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "Make sure the CUDA version of PyTorch is installed."
    )

device = torch.cuda.current_device()

gpu_name = torch.cuda.get_device_name(device)

gpu_memory = (
    torch.cuda.get_device_properties(device).total_memory
    / 1024**3
)

print(f"GPU:  {gpu_name}")
print(f"VRAM: {gpu_memory:.2f} GB")


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded.")

print(f"Vocabulary size: {len(tokenizer):,}")

if tokenizer.chat_template is None:
    raise RuntimeError(
        "The tokenizer does not have a chat template. "
        "A conversational dataset requires a chat template."
    )

print("Chat template: available")


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING 4-BIT QLoRA CONFIGURATION")
print("=" * 60)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    # NF4 is recommended for QLoRA
    bnb_4bit_quant_type="nf4",

    # Saves additional memory
    bnb_4bit_use_double_quant=True,

    # RTX 4060 supports FP16
    bnb_4bit_compute_dtype=torch.float16,
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    # Don't use cache during gradient checkpointing
    torch_dtype=torch.float16,
)

model.config.use_cache = False

print("Model loaded successfully.")

print(f"Model device: {model.device}")


# ============================================================
# PREPARE MODEL FOR QLoRA
# ============================================================

print("\nPreparing model for k-bit training...")

model = prepare_model_for_kbit_training(model)

print("Model prepared.")


# ============================================================
# LoRA CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING LoRA CONFIGURATION")
print("=" * 60)

peft_config = LoraConfig(
    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    # Apply LoRA to the linear layers.
    # This works well with Qwen architectures.
    target_modules="all-linear",
)

print("LoRA configuration created.")


# ============================================================
# LOAD DATASETS
# ============================================================

print("\n" + "=" * 60)
print("LOADING CLEAN DATASETS")
print("=" * 60)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training dataset not found:\n{TRAIN_PATH}"
    )

if not os.path.exists(VALIDATION_PATH):
    raise FileNotFoundError(
        f"Validation dataset not found:\n{VALIDATION_PATH}"
    )

train_dataset = load_from_disk(TRAIN_PATH)

validation_dataset = load_from_disk(
    VALIDATION_PATH
)

print(f"Training examples:   {len(train_dataset):,}")
print(f"Validation examples: {len(validation_dataset):,}")


# ============================================================
# VERIFY DATA FORMAT
# ============================================================

print("\nChecking dataset format...")

if "messages" not in train_dataset.column_names:
    raise RuntimeError(
        "Training dataset does not contain a 'messages' column."
    )

if "messages" not in validation_dataset.column_names:
    raise RuntimeError(
        "Validation dataset does not contain a 'messages' column."
    )

example = train_dataset[0]

print("Dataset format: conversational")
print("Column: messages")

print("\nFirst conversation roles:")

for message in example["messages"]:
    print(f"  {message['role']}")

print("\nDataset format looks good.")


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

print("\n" + "=" * 60)
print("CREATING TRAINING CONFIGURATION")
print("=" * 60)

training_args = SFTConfig(

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=OUTPUT_DIR,

    # --------------------------------------------------------
    # Training duration
    # --------------------------------------------------------

    num_train_epochs=NUM_EPOCHS,

    # IMPORTANT:
    # This limits the first test to only 20 steps.
    max_steps=MAX_STEPS,

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    # Simulates a larger batch while keeping VRAM usage low.
    gradient_accumulation_steps=8,

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    learning_rate=2e-4,

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optim="paged_adamw_8bit",

    # --------------------------------------------------------
    # Sequence length
    # --------------------------------------------------------

    max_length=MAX_LENGTH,

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    gradient_checkpointing=True,

    # IMPORTANT:
    # Packing is disabled for the first test.
    #
    # This avoids the Flash Attention / padding-free warning
    # you saw earlier.
    packing=False,

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    fp16=True,

    bf16=False,

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------

    logging_strategy="steps",

    logging_steps=5,

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    eval_strategy="steps",

    eval_steps=10,

    # --------------------------------------------------------
    # CHECKPOINTS
    # --------------------------------------------------------

    save_strategy="steps",

    save_steps=10,

    save_total_limit=1,

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    # We are using a conversational `messages` dataset.
    #
    # TRL will apply the Qwen chat template and tokenize it.
    dataset_text_field=None,

    # --------------------------------------------------------
    # REPORTING
    # --------------------------------------------------------

    report_to="none",

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed=42,
)


# ============================================================
# CREATE TRAINER
# ============================================================

print("\n" + "=" * 60)
print("CREATING SFT TRAINER")
print("=" * 60)

trainer = SFTTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    peft_config=peft_config,
)

print("SFTTrainer created successfully.")


# ============================================================
# TRAIN
# ============================================================

print("\n")
print("=" * 60)
print("STARTING QLoRA TRAINING")
print("=" * 60)

print(f"Model:        {MODEL_NAME}")
print(f"Training data: {len(train_dataset):,}")
print(f"Validation:    {len(validation_dataset):,}")
print(f"Max length:    {MAX_LENGTH}")
print(f"Batch size:    1")
print(f"Accumulation:  8")
print(f"Max steps:     {MAX_STEPS}")
print("=" * 60)

trainer.train()


# ============================================================
# SAVE
# ============================================================

print("\n" + "=" * 60)
print("SAVING LoRA ADAPTER")
print("=" * 60)

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nAdapter saved to:")
print(OUTPUT_DIR)


# ============================================================
# FINAL GPU INFORMATION
# ============================================================

if torch.cuda.is_available():

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print("\n" + "=" * 60)
    print("GPU MEMORY")
    print("=" * 60)

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved:  {reserved:.2f} GB")


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)
print("20-STEP TEST COMPLETE")
print("=" * 60)

print("\nIf the loss values appeared and there was no CUDA")
print("out-of-memory error, the training pipeline works.")

print("\nNext step:")
print("Remove/disable MAX_STEPS and start the full training run.")