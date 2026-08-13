import os
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
VAL_PATH = "data/processed/validation_clean"

OUTPUT_DIR = "model/checkpoints"
FINAL_DIR = "model/final"

MAX_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

LEARNING_RATE = 2e-4
NUM_EPOCHS = 2

BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    print("=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"GPU:  {gpu}")
    print(f"VRAM: {vram:.2f} GB")

    print("\n" + "=" * 60)
    print("LOADING TOKENIZER")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded.")
    print(f"Vocabulary size: {len(tokenizer)}")

    print("\n" + "=" * 60)
    print("CREATING 4-BIT QLoRA CONFIGURATION")
    print("=" * 60)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    model.config.use_cache = False

    print("Model loaded successfully.")
    print(f"Model device: {model.device}")

    print("\nPreparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)
    print("Model prepared.")

    print("\n" + "=" * 60)
    print("CREATING LoRA CONFIGURATION")
    print("=" * 60)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("LoRA configuration created.")

    print("\n" + "=" * 60)
    print("LOADING CLEAN DATASETS")
    print("=" * 60)

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VAL_PATH)

    print(f"Training examples:   {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")

    print("\nConverting conversations to text...")

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_dataset = train_dataset.map(
        format_example,
        remove_columns=train_dataset.column_names,
        desc="Formatting training dataset",
    )

    validation_dataset = validation_dataset.map(
        format_example,
        remove_columns=validation_dataset.column_names,
        desc="Formatting validation dataset",
    )

    print("Dataset conversion complete.")

    print("\nSample formatted training example:")
    print("-" * 60)
    print(train_dataset[0]["text"][:2000])
    print("-" * 60)

    print("\n" + "=" * 60)
    print("CREATING TRAINING CONFIGURATION")
    print("=" * 60)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION,

        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        logging_steps=10,

        eval_strategy="steps",
        eval_steps=100,

        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,

        fp16=True,
        bf16=False,

        gradient_checkpointing=True,

        max_length=MAX_LENGTH,

        packing=False,

        dataset_text_field="text",

        report_to="none",

        optim="paged_adamw_8bit",

        seed=42,
    )

    print("\n" + "=" * 60)
    print("CREATING SFT TRAINER")
    print("=" * 60)

    trainer = SFTTrainer(
        model=model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    print("\n" + "=" * 60)
    print("SAVING FINAL MODEL")
    print("=" * 60)

    os.makedirs(FINAL_DIR, exist_ok=True)

    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)

    print(f"\nModel saved to: {FINAL_DIR}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()