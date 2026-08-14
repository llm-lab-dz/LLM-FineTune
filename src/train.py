import os

# ============================================================
# IMPORTANT:
# Kaggle Tesla T4 does NOT support BF16 properly.
# Force Accelerate to use FP16.
# This must be set before importing/creating the Trainer.
# ============================================================

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

from trl import (
    SFTConfig,
    SFTTrainer,
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


# ============================================================
# DATA
# ============================================================

TRAIN_PATH = "data/processed/train_clean"

VALIDATION_PATH = "data/processed/validation_clean"


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = "outputs/qwen-3b-qlora"


# ============================================================
# SEQUENCE LENGTH
# ============================================================

MAX_SEQ_LENGTH = 1024


# ============================================================
# LoRA CONFIGURATION
# ============================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

NUM_EPOCHS = 1

LEARNING_RATE = 2e-4


# ============================================================
# GPU CONFIGURATION
# ============================================================

# One sample per GPU
BATCH_SIZE_GPU = 1

# 1 sample × 2 GPUs × 4 accumulation = effective batch 8
GRADIENT_ACCUMULATION_STEPS_GPU = 4


# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(text):

    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


# ============================================================
# DISTRIBUTED GPU INFORMATION
# ============================================================

def get_rank_info():

    world_size = int(
        os.environ.get(
            "WORLD_SIZE",
            "1",
        )
    )

    local_rank = int(
        os.environ.get(
            "LOCAL_RANK",
            "0",
        )
    )

    rank = int(
        os.environ.get(
            "RANK",
            "0",
        )
    )

    return local_rank, rank, world_size


# ============================================================
# FORMAT CONVERSATION
# ============================================================

def format_conversation(
    example,
    tokenizer,
):

    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "text": text
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # RANDOM SEED
    # --------------------------------------------------------

    torch.manual_seed(SEED)

    # --------------------------------------------------------
    # DISTRIBUTED INFORMATION
    # --------------------------------------------------------

    local_rank, rank, world_size = get_rank_info()

    # --------------------------------------------------------
    # CHECK CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU is required for QLoRA training."
        )

    # --------------------------------------------------------
    # SELECT GPU FOR THIS PROCESS
    # --------------------------------------------------------

    torch.cuda.set_device(local_rank)

    device = torch.device(
        f"cuda:{local_rank}"
    )

    # --------------------------------------------------------
    # GPU INFORMATION
    # --------------------------------------------------------

    gpu_name = torch.cuda.get_device_name(
        local_rank
    )

    vram = (
        torch.cuda
        .get_device_properties(local_rank)
        .total_memory
        / (1024 ** 3)
    )

    if rank == 0:

        print_header(
            "DEVICE INFORMATION"
        )

        print(
            f"World size: {world_size}"
        )

        print(
            f"Local rank: {local_rank}"
        )

        print(
            f"GPU:        {gpu_name}"
        )

        print(
            f"VRAM:       {vram:.2f} GB"
        )

        print(
            f"CUDA:       {torch.version.cuda}"
        )

        if world_size == 2:

            print(
                "Using 2 GPUs for distributed training."
            )

        else:

            print(
                f"Using {world_size} GPU."
            )


    # ========================================================
    # TOKENIZER
    # ========================================================

    if rank == 0:

        print_header(
            "LOADING TOKENIZER"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    if rank == 0:

        print(
            "Tokenizer loaded successfully."
        )

        print(
            f"Vocabulary size: {len(tokenizer)}"
        )


    # ========================================================
    # 4-BIT QUANTIZATION
    # ========================================================

    if rank == 0:

        print_header(
            "4-BIT QLoRA CONFIGURATION"
        )

        print(
            "Quantization: 4-bit"
        )

        print(
            "Quantization type: NF4"
        )

        print(
            "Double quantization: True"
        )

        print(
            "Compute dtype: FP16"
        )

    quantization_config = BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_use_double_quant=True,

        bnb_4bit_compute_dtype=torch.float16,
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    if rank == 0:

        print_header(
            "LOADING QWEN 2.5 3B"
        )

        print(
            MODEL_NAME
        )

    model = AutoModelForCausalLM.from_pretrained(

        MODEL_NAME,

        quantization_config=(
            quantization_config
        ),

        # IMPORTANT:
        # Each process gets its own GPU.
        device_map={
            "": local_rank
        },

        dtype=torch.float16,

        trust_remote_code=True,
    )

    if rank == 0:

        print(
            "Model loaded successfully."
        )


    # ========================================================
    # PREPARE MODEL FOR QLoRA
    # ========================================================

    if rank == 0:

        print_header(
            "PREPARING MODEL FOR QLoRA"
        )

    model = prepare_model_for_kbit_training(
        model
    )

    model.config.use_cache = False

    if rank == 0:

        print(
            "Model prepared for k-bit training."
        )


    # ========================================================
    # LoRA CONFIGURATION
    # ========================================================

    if rank == 0:

        print_header(
            "CREATING LoRA CONFIGURATION"
        )

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

    if rank == 0:

        print(
            "LoRA configuration created."
        )

        print(
            f"LoRA rank: {LORA_R}"
        )

        print(
            f"LoRA alpha: {LORA_ALPHA}"
        )

        print(
            f"LoRA dropout: {LORA_DROPOUT}"
        )


    # ========================================================
    # LOAD DATASETS
    # ========================================================

    if rank == 0:

        print_header(
            "LOADING CLEAN DATASETS"
        )

    train_dataset = load_from_disk(
        TRAIN_PATH
    )

    validation_dataset = load_from_disk(
        VALIDATION_PATH
    )

    if rank == 0:

        print(
            f"Training examples:   "
            f"{len(train_dataset):,}"
        )

        print(
            f"Validation examples: "
            f"{len(validation_dataset):,}"
        )


    # ========================================================
    # CONVERT CHAT MESSAGES TO TEXT
    # ========================================================

    if rank == 0:

        print_header(
            "FORMATTING CONVERSATIONS"
        )

    train_dataset = train_dataset.map(

        lambda x: format_conversation(
            x,
            tokenizer,
        ),

        remove_columns=(
            train_dataset.column_names
        ),

        desc="Formatting training dataset",
    )

    validation_dataset = validation_dataset.map(

        lambda x: format_conversation(
            x,
            tokenizer,
        ),

        remove_columns=(
            validation_dataset.column_names
        ),

        desc="Formatting validation dataset",
    )

    if rank == 0:

        print(
            "Dataset formatting complete."
        )

        print()
        print(
            "Example:"
        )

        print(
            "-" * 70
        )

        print(
            train_dataset[0]["text"][:3000]
        )

        print(
            "-" * 70
        )


    # ========================================================
    # TRAINING CONFIGURATION
    # ========================================================

    if rank == 0:

        print_header(
            "CREATING TRAINING CONFIGURATION"
        )

    training_args = SFTConfig(

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_dir=OUTPUT_DIR,

        # ----------------------------------------------------
        # Epochs
        # ----------------------------------------------------

        num_train_epochs=NUM_EPOCHS,

        # ----------------------------------------------------
        # Batch size
        # ----------------------------------------------------

        per_device_train_batch_size=(
            BATCH_SIZE_GPU
        ),

        per_device_eval_batch_size=1,

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS_GPU
        ),

        # ----------------------------------------------------
        # Learning rate
        # ----------------------------------------------------

        learning_rate=LEARNING_RATE,

        lr_scheduler_type="cosine",

        warmup_steps=50,

        # ----------------------------------------------------
        # Gradient clipping
        #
        # Disabled because the T4/BF16 issue occurred
        # during gradient unscaling/clipping.
        # ----------------------------------------------------

        max_grad_norm=0.0,

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_steps=10,

        report_to="none",

        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        save_strategy="steps",

        save_steps=250,

        save_total_limit=2,

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        eval_strategy="steps",

        eval_steps=250,

        # ----------------------------------------------------
        # PRECISION
        #
        # IMPORTANT FOR TESLA T4
        # ----------------------------------------------------

        fp16=True,

        bf16=False,

        # ----------------------------------------------------
        # Gradient checkpointing
        # ----------------------------------------------------

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optim="paged_adamw_8bit",

        # ----------------------------------------------------
        # Sequence length
        # ----------------------------------------------------

        max_length=MAX_SEQ_LENGTH,

        # ----------------------------------------------------
        # Packing
        # ----------------------------------------------------

        packing=False,

        # ----------------------------------------------------
        # Dataset field
        # ----------------------------------------------------

        dataset_text_field="text",

        # ----------------------------------------------------
        # Tokenizer tokens
        # ----------------------------------------------------

        eos_token=tokenizer.eos_token,

        pad_token=tokenizer.pad_token,

        # ----------------------------------------------------
        # Distributed training
        # ----------------------------------------------------

        ddp_find_unused_parameters=False,

        # ----------------------------------------------------
        # Seed
        # ----------------------------------------------------

        seed=SEED,

        # ----------------------------------------------------
        # Dataset columns
        # ----------------------------------------------------

        remove_unused_columns=False,
    )


    # ========================================================
    # CREATE TRAINER
    # ========================================================

    if rank == 0:

        print_header(
            "CREATING SFT TRAINER"
        )

    trainer = SFTTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )


    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    if rank == 0:

        print_header(
            "STARTING QLoRA TRAINING"
        )

        print(
            f"Model:                 "
            f"{MODEL_NAME}"
        )

        print(
            f"GPUs:                  "
            f"{world_size}"
        )

        print(
            f"Batch per GPU:         "
            f"{BATCH_SIZE_GPU}"
        )

        print(
            f"Gradient accumulation: "
            f"{GRADIENT_ACCUMULATION_STEPS_GPU}"
        )

        effective_batch_size = (
            BATCH_SIZE_GPU
            * world_size
            * GRADIENT_ACCUMULATION_STEPS_GPU
        )

        print(
            f"Effective batch size:  "
            f"{effective_batch_size}"
        )

        print(
            f"Learning rate:         "
            f"{LEARNING_RATE}"
        )

        print(
            f"Max sequence length:   "
            f"{MAX_SEQ_LENGTH}"
        )

        print(
            f"Epochs:                "
            f"{NUM_EPOCHS}"
        )

        print(
            f"Output directory:      "
            f"{OUTPUT_DIR}"
        )

        print()
        print(
            "Precision: FP16"
        )

        print(
            "Quantization: 4-bit NF4"
        )

        print(
            "Method: QLoRA"
        )

        print(
            "Starting training..."
        )


    # ========================================================
    # TRAIN
    # ========================================================

    trainer.train()


    # ========================================================
    # SAVE MODEL
    # ========================================================

    if rank == 0:

        print_header(
            "SAVING MODEL"
        )

        trainer.save_model(
            OUTPUT_DIR
        )

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )

        print()
        print(
            "LoRA adapter saved to:"
        )

        print(
            OUTPUT_DIR
        )

        print_header(
            "TRAINING COMPLETE"
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()