import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "llm-lab-dz/qwen-3b-qlora-ultrachat"

USE_GPU = torch.cuda.is_available()

TEST_PROMPTS = [
    "Explain what a GPU is in simple terms.",
    "Write a short poem about autumn.",
    "What's the difference between a list and a tuple in Python?",
]


def load_base_model(tokenizer):
    if USE_GPU:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float32,
            device_map="cpu",
        )
    return model


def generate(model, tokenizer, prompt, max_new_tokens=150):
    messages = [{"role": "user", "content": prompt}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
        )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return response


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Loading base model...")
    base_model = load_base_model(tokenizer)

    print(f"Loading fine-tuned adapter from {ADAPTER_REPO}...")
    finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)

    print("\nModels loaded. Running comparisons...\n")

    for prompt in TEST_PROMPTS:
        print("=" * 70)
        print(f"PROMPT: {prompt}")
        print("=" * 70)

        print("\n--- BASE MODEL ---")
        base_response = generate(base_model, tokenizer, prompt)
        print(base_response)

        print("\n--- FINE-TUNED MODEL ---")
        finetuned_response = generate(finetuned_model, tokenizer, prompt)
        print(finetuned_response)

        print()


if __name__ == "__main__":
    main()