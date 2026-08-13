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
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_PATH = "data/processed/train_clean"
VALIDATION_PATH = "data/processed/validation_clean"

OUTPUT_DIR = "outputs/qwen-3b-qlora"

MAX_SEQ_LENGTH = 1024

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

NUM_EPOCHS = 1
LEARNING_RATE = 2e-4

BATCH_SIZE_GPU = 1
BATCH_SIZE_CPU = 1

GRADIENT_ACCUMULATION_STEPS_GPU = 8
GRADIENT_ACCUMULATION_STEPS_CPU = 2

SEED = 42


def print_header(text):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def format_conversation(example, tokenizer):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


def main():

    torch.manual_seed(SEED)

    device = get_device()

    print_header("DEVICE INFORMATION")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        print(f"Device: GPU")
        print(f"GPU:    {gpu_name}")
        print(f"VRAM:   {vram:.2f} GB")

        use_4bit = True
        use_fp16 = False
        use_bf16 = False

        batch_size = BATCH_SIZE_GPU
        gradient_accumulation = GRADIENT_ACCUMULATION_STEPS_GPU

    else:
        print("Device: CPU")
        print("No CUDA GPU detected.")

        use_4bit = False
        use_fp16 = False
        use_bf16 = False

        batch_size = BATCH_SIZE_CPU
        gradient_accumulation = GRADIENT_ACCUMULATION_STEPS_CPU

    print_header("LOADING TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded.")
    print(f"Vocabulary size: {len(tokenizer)}")

    print_header("MODEL CONFIGURATION")

    quantization_config = None

    if use_4bit:

        print("Using 4-bit NF4 QLoRA.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    else:

        print("Using CPU mode.")
        print("4-bit quantization disabled.")

    print_header("LOADING MODEL")

    model_kwargs = {
        "trust_remote_code": True,
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": 0}

    else:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    print("Model loaded successfully.")

    if device == "cuda":

        print("Preparing model for k-bit training...")

        model = prepare_model_for_kbit_training(model)

        print("Model prepared.")

    else:

        model.gradient_checkpointing_enable()

    model.config.use_cache = False

    print_header("CREATING LoRA CONFIGURATION")

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

    print_header("LOADING CLEAN DATASETS")

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)

    print(f"Training examples:   {len(train_dataset):,}")
    print(f"Validation examples: {len(validation_dataset):,}")

    print_header("CONVERTING CONVERSATIONS TO TEXT")

    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="Formatting training dataset",
    )

    validation_dataset = validation_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=validation_dataset.column_names,
        desc="Formatting validation dataset",
    )

    print("Dataset conversion complete.")

    print()
    print("Sample:")
    print("-" * 60)
    print(train_dataset[0]["text"][:3000])
    print("-" * 60)

    print_header("CREATING TRAINING CONFIGURATION")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=gradient_accumulation,

        learning_rate=LEARNING_RATE,

        logging_steps=10,

        save_steps=250,
        save_total_limit=2,

        eval_strategy="steps",
        eval_steps=250,

        fp16=use_fp16,
        bf16=use_bf16,

        gradient_checkpointing=True,

        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",

        max_grad_norm=1.0,

        lr_scheduler_type="cosine",

        warmup_steps=50,

        report_to="none",

        seed=SEED,

        max_length=MAX_SEQ_LENGTH,

        packing=False,

        dataset_text_field="text",

        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,

        remove_unused_columns=False,
    )

    print_header("CREATING SFT TRAINER")

    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    print_header("STARTING TRAINING")

    print(f"Epochs:                 {NUM_EPOCHS}")
    print(f"Batch size:             {batch_size}")
    print(f"Gradient accumulation:  {gradient_accumulation}")
    print(f"Effective batch size:   {batch_size * gradient_accumulation}")
    print(f"Learning rate:          {LEARNING_RATE}")
    print(f"Max sequence length:    {MAX_SEQ_LENGTH}")
    print(f"Output directory:       {OUTPUT_DIR}")

    trainer.train()

    print_header("SAVING MODEL")

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Model saved to:")
    print(OUTPUT_DIR)

    print_header("TRAINING COMPLETE")


if __name__ == "__main__":
    main()