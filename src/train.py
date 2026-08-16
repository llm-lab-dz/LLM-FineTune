import os

os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"

import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
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

SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 3

SEED = 42

# ============================================================
# Hugging Face Hub settings
# ============================================================

ENABLE_HUB_PUSH = os.getenv("ENABLE_HUB_PUSH", "false").lower() == "true"
HF_REPO_ID = os.getenv("HF_REPO_ID")
HF_PRIVATE_REPO = os.getenv("HF_PRIVATE_REPO", "true").lower() == "true"
PUSH_TO_HUB_EVERY_N_SAVES = int(os.getenv("PUSH_TO_HUB_EVERY_N_SAVES", "5"))


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_conversation(example, tokenizer):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def find_latest_checkpoint():
    if not os.path.exists(OUTPUT_DIR):
        return None

    checkpoints = []

    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)

        if os.path.isdir(path) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, path))
            except ValueError:
                pass

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])

    return checkpoints[-1][1]


def push_folder_to_hub(folder_path, repo_id, commit_message):
    """
    Upload a folder to the Hugging Face Hub.
    Never lets a Hub failure crash the training run.
    """
    if not ENABLE_HUB_PUSH or not repo_id:
        print(f"[hub] Skipped upload for '{folder_path}'. ENABLE_HUB_PUSH is off or no HF_REPO_ID is configured.")
        return

    from huggingface_hub import HfApi

    try:
        api = HfApi()

        api.create_repo(
            repo_id,
            exist_ok=True,
            private=HF_PRIVATE_REPO,
        )

        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )

        print(f"[hub] Pushed '{folder_path}' -> https://huggingface.co/{repo_id}")

    except Exception as e:
        print(f"[hub] WARNING: push failed, continuing training. Error: {e}")


class PushToHubCallback(TrainerCallback):
    """
    Periodically pushes checkpoints to the Hugging Face Hub during training,
    so a disconnected/expired session doesn't lose progress.
    """

    def __init__(self, repo_id, every_n_saves):
        self.repo_id = repo_id
        self.every_n_saves = every_n_saves
        self.save_count = 0

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.save_count += 1

            if self.save_count % self.every_n_saves == 0:
                checkpoint_dir = os.path.join(
                    args.output_dir,
                    f"checkpoint-{state.global_step}",
                )

                if os.path.isdir(checkpoint_dir):
                    push_folder_to_hub(
                        checkpoint_dir,
                        self.repo_id,
                        commit_message=f"Checkpoint at step {state.global_step}",
                    )

        return control


def main():

    torch.manual_seed(SEED)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    torch.cuda.set_device(local_rank)

    if rank == 0:
        section("DEVICE")

        print("GPUs:", world_size)

        for i in range(torch.cuda.device_count()):
            memory = (
                torch.cuda.get_device_properties(i).total_memory
                / 1024**3
            )

            print(
                f"GPU {i}: "
                f"{torch.cuda.get_device_name(i)} "
                f"({memory:.2f} GB)"
            )

        print("PyTorch:", torch.__version__)
        print("CUDA:", torch.version.cuda)
        print("Mixed precision: fp16")

    section("TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        print("Tokenizer loaded")
        print("Vocabulary:", len(tokenizer))

    section("QLORA")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    if rank == 0:
        print("4-bit: True")
        print("NF4: True")
        print("Double quantization: True")
        print("Compute dtype: FP16")

    section("MODEL")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map={"": local_rank},
        dtype=torch.float16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.config.torch_dtype = torch.float16

    if rank == 0:
        print("Model loaded")
        print("Model dtype:", model.dtype)

    section("LORA")

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
        print("LoRA rank:", LORA_R)
        print("LoRA alpha:", LORA_ALPHA)
        print("LoRA dropout:", LORA_DROPOUT)

    section("DATA")

    train_dataset = load_from_disk(TRAIN_PATH)
    validation_dataset = load_from_disk(VALIDATION_PATH)

    if rank == 0:
        print("Train:", len(train_dataset))
        print("Validation:", len(validation_dataset))

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
        print("Dataset formatting complete")

    section("TRAINING CONFIG")

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        num_train_epochs=NUM_EPOCHS,

        per_device_train_batch_size=BATCH_SIZE_GPU,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,

        logging_steps=10,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        eval_strategy="steps",
        eval_steps=250,

        fp16=True,
        bf16=False,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
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
        print("Epochs:", NUM_EPOCHS)
        print("Batch/GPU:", BATCH_SIZE_GPU)
        print(
            "Gradient accumulation:",
            GRADIENT_ACCUMULATION_STEPS
        )

        print(
            "Effective batch:",
            BATCH_SIZE_GPU
            * world_size
            * GRADIENT_ACCUMULATION_STEPS
        )

        print("Learning rate:", LEARNING_RATE)
        print("Max sequence length:", MAX_SEQ_LENGTH)

        print("FP16: True")
        print("BF16: False")

        print("Checkpoint every:", SAVE_STEPS, "steps")
        print("Maximum checkpoints:", SAVE_TOTAL_LIMIT)

        print("Hub repo:", HF_REPO_ID)
        print("Push to Hub every:", PUSH_TO_HUB_EVERY_N_SAVES, "saves")

    section("SFT TRAINER")

    callbacks = []

    if rank == 0 and ENABLE_HUB_PUSH and HF_REPO_ID:
        callbacks.append(
            PushToHubCallback(
                repo_id=HF_REPO_ID,
                every_n_saves=PUSH_TO_HUB_EVERY_N_SAVES,
            )
        )
    elif rank == 0:
        print("[hub] Hub uploads are disabled. Local training only.")

    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=train_dataset,

        eval_dataset=validation_dataset,

        processing_class=tokenizer,

        peft_config=lora_config,

        callbacks=callbacks,
    )

    if rank == 0:
        print("SFTTrainer created")

        trainer.model.print_trainable_parameters()

    section("CHECKPOINT")

    latest_checkpoint = find_latest_checkpoint()

    if rank == 0:

        if latest_checkpoint:

            print("Checkpoint found:")
            print(latest_checkpoint)

            print()
            print("Training will resume from this checkpoint.")

        else:

            print("No checkpoint found.")

            print()
            print("Training will start from the beginning.")

    section("TRAINING")

    if rank == 0:
        print("Starting QLoRA training...")

    try:
        trainer.train(
            resume_from_checkpoint=latest_checkpoint
        )
    except Exception as e:
        # If training crashes partway through, still try to push
        # whatever the latest checkpoint is before re-raising.
        if rank == 0:
            section("TRAINING INTERRUPTED — ATTEMPTING EMERGENCY SAVE")
            print(f"Error: {e}")

            emergency_checkpoint = find_latest_checkpoint()

            if emergency_checkpoint:
                push_folder_to_hub(
                    emergency_checkpoint,
                    HF_REPO_ID,
                    commit_message="Emergency save after training interruption",
                )
        raise

    if rank == 0:

        section("SAVING FINAL MODEL")

        trainer.save_model(OUTPUT_DIR)

        tokenizer.save_pretrained(OUTPUT_DIR)

        print("Final model saved to:")
        print(OUTPUT_DIR)

        section("PUSHING FINAL MODEL TO HUGGING FACE HUB")

        push_folder_to_hub(
            OUTPUT_DIR,
            HF_REPO_ID,
            commit_message="Final model after training completion",
        )

        section("DONE")


if __name__ == "__main__":
    main()