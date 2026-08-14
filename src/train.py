import os

os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

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
GRADIENT_ACCUMULATION_STEPS = 4

SEED = 42


def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_rank(message, rank):
    if rank == 0:
        print(message)


def format_conversation(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    torch.manual_seed(SEED)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training.")

    torch.cuda.set_device(local_rank)

    print_section(f"PROCESS {rank} / {world_size}")

    print(f"Local rank:       {local_rank}")
    print(f"Global rank:      {rank}")
    print(f"World size:       {world_size}")
    print(f"GPU:              {torch.cuda.get_device_name(local_rank)}")
    print(f"CUDA:             {torch.version.cuda}")
    print(f"PyTorch:          {torch.__version__}")
    print(
        f"GPU memory:       "
        f"{torch.cuda.get_device_properties(local_rank).total_memory / 1024**3:.2f} GB"
    )
    print(f"Mixed precision:  {os.environ.get('ACCELERATE_MIXED_PRECISION')}")

    print_section("TRAINING CONFIGURATION")

    if rank == 0:
        print(f"Model:             {MODEL_NAME}")
        print(f"Train dataset:     {TRAIN_PATH}")
        print(f"Validation dataset:{VALIDATION_PATH}")
        print(f"Output:            {OUTPUT_DIR}")
        print(f"Sequence length:   {MAX_SEQ_LENGTH}")
        print(f"Epochs:            {NUM_EPOCHS}")
        print(f"Learning rate:     {LEARNING_RATE}")
        print(f"Batch/GPU:         {BATCH_SIZE_GPU}")
        print(f"Grad accumulation: {GRADIENT_ACCUMULATION_STEPS}")
        print(f"Effective batch:   {BATCH_SIZE_GPU * world_size * GRADIENT_ACCUMULATION_STEPS}")
        print(f"LoRA rank:         {LORA_R}")
        print(f"LoRA alpha:        {LORA_ALPHA}")
        print(f"LoRA dropout:      {LORA_DROPOUT}")

    print_section("LOADING TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print_rank(
        f"Tokenizer loaded | vocab size: {len(tokenizer)}",
        rank,
    )

    print_section("QLORA CONFIGURATION")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if rank == 0:
        print("4-bit quantization:     enabled")
        print("Quantization type:      NF4")
        print("Double quantization:    enabled")
        print("Compute dtype:          float16")

    print_section("LOADING MODEL")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": local_rank},
        dtype=torch.float16,
        trust_remote_code=True,
    )

    print_rank("Model loaded.", rank)

    print_section("FORCING FP16")

    model.config.torch_dtype = torch.float16
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float16)

    print_rank("Model prepared for k-bit training.", rank)

    print_section("LORA CONFIGURATION")

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

    print_rank("LoRA configuration ready.", rank)

    print_section("CHECKING TRAINABLE PARAMETERS")

    trainable_params = 0
    all_params = 0

    first_trainable_name = None
    first_trainable_dtype = None

    for name, param in model.named_parameters():
        all_params += param.numel()

        if param.requires_grad:
            trainable_params += param.numel()

            if first_trainable_name is None:
                first_trainable_name = name
                first_trainable_dtype = param.dtype

    if rank == 0:
        print(f"Total parameters:      {all_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(
            f"Trainable percentage:  "
            f"{100 * trainable_params / all_params:.4f}%"
        )
        print(f"First trainable tensor: {first_trainable_name}")
        print(f"Trainable dtype:        {first_trainable_dtype}")

    if first_trainable_dtype != torch.float16:
        raise RuntimeError(
            f"Expected trainable parameters to be FP16, "
            f"but found {first_trainable_dtype}"
        )

    print_rank(
        "Trainable parameter dtype check: FP16 OK",
        rank,
    )

    print_section("LOADING DATASETS")

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)

    if rank == 0:
        print(f"Training examples:   {len(train_dataset):,}")
        print(f"Validation examples: {len(validation_dataset):,}")

    print_section("FORMATTING DATASETS")

    train_dataset = train_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=train_dataset.column_names,
        desc="Formatting train",
    )

    validation_dataset = validation_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=validation_dataset.column_names,
        desc="Formatting validation",
    )

    if rank == 0:
        print("Dataset formatting complete.")
        print()
        print("Sample:")
        print("-" * 70)
        print(train_dataset[0]["text"][:2000])
        print("-" * 70)

    print_section("SFT CONFIGURATION")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE_GPU,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=250,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },
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

    if rank == 0:
        print("FP16:               True")
        print("BF16:               False")
        print("Optimizer:          paged_adamw_8bit")
        print("Gradient checkpoint: True")
        print("Max grad norm:      0.0")

    print_section("CREATING SFT TRAINER")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print_rank(
        "SFTTrainer created successfully.",
        rank,
    )

    print_section("STARTING TRAINING")

    if rank == 0:
        print("Everything passed the checks.")
        print("Starting QLoRA training now...")
        print()

    trainer.train()

    print_section("SAVING MODEL")

    if rank == 0:
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)

        print(f"Saved to: {OUTPUT_DIR}")

    print_section("TRAINING COMPLETE")

    if rank == 0:
        print("QLoRA training finished successfully.")


if __name__ == "__main__":
    main()