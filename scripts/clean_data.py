from datasets import load_from_disk


MIN_TURNS = 2
MIN_MSG_CHARS = 3
MAX_TOTAL_CHARS = 12_000
MIN_TOTAL_CHARS = 20

TRAIN_INPUT = "data/processed/train"
TRAIN_OUTPUT = "data/processed/train_clean"

VALIDATION_INPUT = "data/processed/validation"
VALIDATION_OUTPUT = "data/processed/validation_clean"


def is_clean(example):
    messages = example.get("messages")

    if not isinstance(messages, list):
        return False

    if len(messages) < MIN_TURNS:
        return False

    expected_role = "user"
    total_chars = 0

    for message in messages:
        if not isinstance(message, dict):
            return False

        role = message.get("role")
        content = message.get("content")

        if not isinstance(content, str):
            return False

        content = content.strip()

        if len(content) < MIN_MSG_CHARS:
            return False

        if role not in {"user", "assistant"}:
            return False

        if role != expected_role:
            return False

        if expected_role == "user":
            expected_role = "assistant"
        else:
            expected_role = "user"

        total_chars += len(content)

    if total_chars < MIN_TOTAL_CHARS:
        return False

    if total_chars > MAX_TOTAL_CHARS:
        return False

    if messages[-1].get("role") != "assistant":
        return False

    return True


def clean_example(example):
    cleaned_messages = []

    for message in example["messages"]:
        cleaned_messages.append(
            {
                "role": message["role"],
                "content": message["content"].strip(),
            }
        )

    example["messages"] = cleaned_messages

    return example


def conversation_signature(example):
    return tuple(
        (
            message["role"],
            message["content"].strip(),
        )
        for message in example["messages"]
    )


def remove_duplicates(dataset):
    seen = set()
    keep_indices = []

    for index, example in enumerate(dataset):
        signature = conversation_signature(example)

        if signature not in seen:
            seen.add(signature)
            keep_indices.append(index)

    return dataset.select(keep_indices)


def process_dataset(input_path, output_path, name):
    print("\n" + "=" * 60)
    print(f"Processing {name}")
    print("=" * 60)

    print(f"\nLoading: {input_path}")

    dataset = load_from_disk(input_path)

    original_count = len(dataset)

    print(f"Original examples: {original_count:,}")

    print("\nFiltering conversations...")

    dataset = dataset.filter(is_clean)

    after_filter = len(dataset)
    removed_filter = original_count - after_filter

    print(f"After filtering:   {after_filter:,}")
    print(f"Removed:           {removed_filter:,}")

    print("\nNormalizing whitespace...")

    dataset = dataset.map(
        clean_example,
        desc="Cleaning messages",
    )

    print("\nRemoving exact duplicates...")

    before_dedup = len(dataset)

    dataset = remove_duplicates(dataset)

    after_dedup = len(dataset)
    removed_duplicates = before_dedup - after_dedup

    print(f"After dedup:       {after_dedup:,}")
    print(f"Duplicates removed: {removed_duplicates:,}")

    print(f"\nSaving cleaned {name} dataset...")

    dataset.save_to_disk(output_path)

    print(f"Saved to: {output_path}")
    print(f"Final examples: {len(dataset):,}")

    return dataset


def show_examples(dataset, number=2):
    print("\n" + "=" * 60)
    print("CLEANED DATASET EXAMPLES")
    print("=" * 60)

    for i in range(min(number, len(dataset))):
        print(f"\n--- Example {i + 1} ---")

        for message in dataset[i]["messages"]:
            role = message["role"].upper()
            content = message["content"]

            print(f"\n[{role}]")
            print(content)


def main():
    print("=" * 60)
    print("UltraChat Dataset Cleaner")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Minimum turns:       {MIN_TURNS}")
    print(f"  Minimum message:     {MIN_MSG_CHARS} characters")
    print(f"  Minimum conversation: {MIN_TOTAL_CHARS} characters")
    print(f"  Maximum conversation: {MAX_TOTAL_CHARS} characters")

    train = process_dataset(
        TRAIN_INPUT,
        TRAIN_OUTPUT,
        "TRAIN",
    )

    validation = process_dataset(
        VALIDATION_INPUT,
        VALIDATION_OUTPUT,
        "VALIDATION",
    )

    show_examples(train, number=2)

    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"\nTraining examples:   {len(train):,}")
    print(f"Validation examples: {len(validation):,}")

    print("\nOutput directories:")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {VALIDATION_OUTPUT}")

    print("\nYour cleaned dataset is ready for the next step: QLoRA training.")


if __name__ == "__main__":
    main()