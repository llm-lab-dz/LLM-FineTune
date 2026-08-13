from datasets import load_dataset

DATASET_NAME = "HuggingFaceH4/ultrachat_200k"

print("Loading UltraChat...")

dataset = load_dataset(
    DATASET_NAME,
    split="train_sft"
)

print(f"Full dataset: {len(dataset):,} examples")

# First 10,000 examples for our initial experiment
dataset = dataset.select(range(10_000))

# Shuffle so we don't accidentally train on an ordered subset
dataset = dataset.shuffle(seed=42)

# 90/10 train/validation split
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

print("Dataset prepared successfully.")