import os
import torch

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
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
GRADIENT_ACCUMULATION_STEPS_GPU = 4

SEED = 42


def format_conversation(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def main():
    torch.manual_seed(SEED)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"GPUs: {world_size}")
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": local_rank},
        dtype=torch.float32,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

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

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)

    if rank == 0:
        print(f"Train: {len(train_dataset)}")
        print(f"Validation: {len(validation_dataset)}")

    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )

    validation_dataset = validation_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=validation_dataset.column_names,
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_GPU,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_GPU,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=250,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.0,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        report_to="none",
        seed=SEED,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_text_field="text",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    if rank == 0:
        print("Starting training...")

    trainer.train()

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()