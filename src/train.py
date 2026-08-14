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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

TRAIN_PATH = "data/processed/train_clean"
VALIDATION_PATH = "data/processed/validation_clean"

OUTPUT_DIR = "outputs/qwen-3b-qlora"

MAX_SEQ_LENGTH = 1024

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4

# 2 x T4
BATCH_SIZE_GPU = 1

# With 2 GPUs:
# 1 sample/GPU × 2 GPUs × 4 accumulation = effective batch 8
GRADIENT_ACCUMULATION_STEPS_GPU = 4

SEED = 42


# ============================================================
# HELPERS
# ============================================================

def print_header(text):
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def get_rank_info():

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    return local_rank, rank, world_size


def format_conversation(example, tokenizer):

    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}


# ============================================================
# MAIN
# ============================================================

def main():

    torch.manual_seed(SEED)

    local_rank, rank, world_size = get_rank_info()

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for QLoRA training."
        )

    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")

    gpu_name = torch.cuda.get_device_name(local_rank)

    vram = (
        torch.cuda.get_device_properties(local_rank).total_memory
        / (1024 ** 3)
    )

    # Only rank 0 prints general information
    if rank == 0:

        print_header("DEVICE INFORMATION")

        print(f"World size: {world_size}")
        print(f"Local rank: {local_rank}")
        print(f"GPU:        {gpu_name}")
        print(f"VRAM:       {vram:.2f} GB")

        if world_size == 2:
            print("Using 2 GPUs for distributed training.")
        else:
            print("Using 1 GPU.")

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    if rank == 0:
        print_header("LOADING TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print("Tokenizer loaded.")
        print(f"Vocabulary size: {len(tokenizer)}")

    # --------------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------------

    if rank == 0:
        print_header("4-BIT QLoRA CONFIGURATION")

        print("Using:")
        print("  - 4-bit quantization")
        print("  - NF4")
        print("  - Double quantization")
        print("  - FP16 compute")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    if rank == 0:
        print_header("LOADING QWEN 2.5 3B")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,

        quantization_config=quantization_config,

        # IMPORTANT:
        # Each distributed process loads the model
        # on its own GPU.
        device_map={"": local_rank},

        torch_dtype=torch.float16,

        trust_remote_code=True,
    )

    if rank == 0:
        print("Model loaded successfully.")

    # --------------------------------------------------------
    # PREPARE FOR QLoRA
    # --------------------------------------------------------

    if rank == 0:
        print_header("PREPARING MODEL FOR QLoRA")

    model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    if rank == 0:
        print("Model prepared.")

    # --------------------------------------------------------
    # LoRA
    # --------------------------------------------------------

    if rank == 0:
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

    if rank == 0:
        print("LoRA configuration created.")

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    if rank == 0:
        print_header("LOADING CLEAN DATASETS")

    train_dataset = load_from_disk(
        TRAIN_PATH
    )

    validation_dataset = load_from_disk(
        VALIDATION_PATH
    )

    if rank == 0:

        print(
            f"Training examples:   {len(train_dataset):,}"
        )

        print(
            f"Validation examples: {len(validation_dataset):,}"
        )

    # --------------------------------------------------------
    # CONVERT CHAT → TEXT
    # --------------------------------------------------------

    if rank == 0:
        print_header("FORMATTING CONVERSATIONS")

    train_dataset = train_dataset.map(
        lambda x: format_conversation(
            x,
            tokenizer,
        ),
        remove_columns=train_dataset.column_names,
        desc="Formatting training dataset",
    )

    validation_dataset = validation_dataset.map(
        lambda x: format_conversation(
            x,
            tokenizer,
        ),
        remove_columns=validation_dataset.column_names,
        desc="Formatting validation dataset",
    )

    if rank == 0:

        print("Dataset formatting complete.")

        print()
        print("Example:")
        print("-" * 70)

        print(
            train_dataset[0]["text"][:3000]
        )

        print("-" * 70)

    # --------------------------------------------------------
    # TRAINING CONFIG
    # --------------------------------------------------------

    if rank == 0:
        print_header("CREATING TRAINING CONFIGURATION")

    training_args = SFTConfig(

        output_dir=OUTPUT_DIR,

        # Epochs
        num_train_epochs=NUM_EPOCHS,

        # Batch
        per_device_train_batch_size=BATCH_SIZE_GPU,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS_GPU
        ),

        # Learning
        learning_rate=LEARNING_RATE,

        lr_scheduler_type="cosine",

        warmup_steps=50,

        max_grad_norm=1.0,

        # Logging
        logging_steps=10,

        report_to="none",

        # Checkpoints
        save_strategy="steps",

        save_steps=250,

        save_total_limit=2,

        # Evaluation
        eval_strategy="steps",

        eval_steps=250,

        # Precision
        fp16=True,

        bf16=False,

        # Memory
        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        # Optimizer
        optim="paged_adamw_8bit",

        # Sequence
        max_length=MAX_SEQ_LENGTH,

        packing=False,

        # Dataset
        dataset_text_field="text",

        # Tokens
        eos_token=tokenizer.eos_token,

        pad_token=tokenizer.pad_token,

        # Distributed training
        ddp_find_unused_parameters=False,

        # Reproducibility
        seed=SEED,

        # Don't let Trainer remove our text column
        remove_unused_columns=False,
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    if rank == 0:
        print_header("CREATING SFT TRAINER")

    trainer = SFTTrainer(

        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if rank == 0:

        print_header("STARTING QLoRA TRAINING")

        print(f"Model:                  {MODEL_NAME}")

        print(f"GPUs:                   {world_size}")

        print(
            f"Batch/GPU:              {BATCH_SIZE_GPU}"
        )

        print(
            f"Gradient accumulation:  "
            f"{GRADIENT_ACCUMULATION_STEPS_GPU}"
        )

        print(
            f"Effective batch size:   "
            f"{BATCH_SIZE_GPU * world_size * GRADIENT_ACCUMULATION_STEPS_GPU}"
        )

        print(
            f"Learning rate:          {LEARNING_RATE}"
        )

        print(
            f"Max sequence length:    {MAX_SEQ_LENGTH}"
        )

        print(
            f"Epochs:                 {NUM_EPOCHS}"
        )

        print(
            f"Output directory:       {OUTPUT_DIR}"
        )

    trainer.train()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if rank == 0:

        print_header("SAVING MODEL")

        trainer.save_model(
            OUTPUT_DIR
        )

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )

        print(
            f"LoRA adapter saved to:"
        )

        print(
            OUTPUT_DIR
        )

        print_header("TRAINING COMPLETE")


if __name__ == "__main__":
    main()