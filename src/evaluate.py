import os
import math

import torch

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import SFTConfig, SFTTrainer


MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

VALIDATION_PATH = os.getenv("VALIDATION_PATH", "data/processed/validation_clean")

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))

NUM_SAMPLE_GENERATIONS = int(os.getenv("NUM_SAMPLE_GENERATIONS", "3"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "150"))

SEED = int(os.getenv("SEED", "42"))

# ============================================================
# Hugging Face Hub settings
# ============================================================

HF_REPO_ID = os.getenv("HF_REPO_ID")


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


def load_model_and_tokenizer(use_gpu):
    section("TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded")
    print("Vocabulary:", len(tokenizer))

    section("MODEL")

    if use_gpu:
        print("GPU detected — loading base model in 4-bit.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("No GPU detected — loading base model in full precision on CPU.")

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

    print("Base model loaded")

    section("ADAPTER")

    if not HF_REPO_ID:
        raise ValueError(
            "HF_REPO_ID is not set. Set it as an environment variable "
            "before running evaluate.py, e.g.:\n"
            "  export HF_REPO_ID=your-org/your-model-repo"
        )

    print("Loading fine-tuned adapter:", HF_REPO_ID)

    model = PeftModel.from_pretrained(base_model, HF_REPO_ID)
    model.eval()

    print("Model + adapter loaded")

    return model, tokenizer


def compute_quantitative_metrics(model, tokenizer, use_gpu):
    section("QUANTITATIVE EVALUATION")

    validation_dataset = load_from_disk(VALIDATION_PATH)
    print("Validation examples:", len(validation_dataset))

    validation_dataset = validation_dataset.map(
        lambda x: format_conversation(x, tokenizer),
        remove_columns=validation_dataset.column_names,
        desc="Formatting validation",
    )

    eval_args = SFTConfig(
        output_dir="eval_tmp",

        per_device_eval_batch_size=1,

        report_to="none",

        seed=SEED,

        max_length=MAX_SEQ_LENGTH,

        packing=False,

        dataset_text_field="text",

        remove_unused_columns=False,

        fp16=use_gpu,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=eval_args,
        train_dataset=validation_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    print("\nRunning evaluation over validation set...")

    metrics = trainer.evaluate()

    eval_loss = metrics.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None else None

    print("\n--- Results ---")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    if perplexity is not None:
        print(f"perplexity: {perplexity:.4f}")

    return metrics, perplexity


def run_sample_generations(model, tokenizer):
    section("QUALITATIVE SAMPLES")

    validation_dataset = load_from_disk(VALIDATION_PATH)

    sample_indices = range(min(NUM_SAMPLE_GENERATIONS, len(validation_dataset)))

    for i in sample_indices:
        example = validation_dataset[i]
        messages = example["messages"]

        # Use the first user turn as the prompt, ignore the rest of the
        # ground-truth conversation for this comparison.
        first_user_message = next(
            (m["content"] for m in messages if m["role"] == "user"),
            None,
        )

        if first_user_message is None:
            continue

        print(f"\n--- Sample {i + 1} ---")
        print(f"PROMPT:\n{first_user_message}\n")

        prompt_messages = [{"role": "user", "content": first_user_message}]

        text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Ground-truth assistant reply from the dataset, for comparison.
        first_assistant_message = next(
            (m["content"] for m in messages if m["role"] == "assistant"),
            None,
        )

        print(f"MODEL RESPONSE:\n{response}\n")

        if first_assistant_message:
            print(f"REFERENCE (from dataset):\n{first_assistant_message[:500]}")

            if len(first_assistant_message) > 500:
                print("... (truncated)")


def main():

    torch.manual_seed(SEED)

    use_gpu = torch.cuda.is_available()

    section("DEVICE")

    if use_gpu:
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "GB",
        )
    else:
        print("No GPU detected. Running on CPU.")

    print("PyTorch:", torch.__version__)

    section("CONFIG")

    print("Base model:", MODEL_NAME)
    print("Adapter repo:", HF_REPO_ID)
    print("Validation path:", VALIDATION_PATH)
    print("Max sequence length:", MAX_SEQ_LENGTH)
    print("Sample generations:", NUM_SAMPLE_GENERATIONS)
    print("Max new tokens:", MAX_NEW_TOKENS)

    model, tokenizer = load_model_and_tokenizer(use_gpu)

    metrics, perplexity = compute_quantitative_metrics(model, tokenizer, use_gpu)

    run_sample_generations(model, tokenizer)

    section("SUMMARY")

    print("Adapter:", HF_REPO_ID)
    print("Validation examples:", len(load_from_disk(VALIDATION_PATH)))
    print("Eval loss:", metrics.get("eval_loss"))

    if perplexity is not None:
        print(f"Perplexity: {perplexity:.4f}")

    print("Mean token accuracy:", metrics.get("eval_mean_token_accuracy"))

    section("DONE")


if __name__ == "__main__":
    main()