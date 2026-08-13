from datasets import load_dataset

DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

print("Loading UltraChat...")

dataset = load_dataset(
    DATASET_NAME,
    split="train_sft"
)

print(f"Full dataset: {len(dataset):,} examples")

dataset = dataset.select(range(10_000))
dataset = dataset.shuffle(seed=42)

split = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train = split["train"]
validation = split["test"]

print(f"Training:   {len(train):,}")
print(f"Validation: {len(validation):,}")

train.save_to_disk("data/processed/train")
validation.save_to_disk("data/processed/validation")

print("\nDataset prepared successfully.")

# Show examples
print("\n========== EXAMPLE 1 ==========")

for message in train[0]["messages"]:
    print(f"\n[{message['role'].upper()}]")
    print(message["content"])

print("\n========== EXAMPLE 2 ==========")

for message in train[1]["messages"]:
    print(f"\n[{message['role'].upper()}]")
    print(message["content"])

print("\n========== EXAMPLE 3 ==========")

for message in train[2]["messages"]:
    print(f"\n[{message['role'].upper()}]")
    print(message["content"])