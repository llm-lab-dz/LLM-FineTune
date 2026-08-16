# Project Guide

This document explains the training flow and how to operate the repository end-to-end.

## Repository overview

The project is organized around a simple LLM fine-tuning pipeline:

- dataset preparation
- dataset cleaning
- QLoRA training
- checkpoint export
- local inference and comparison

## Data pipeline

### 1. Download raw data

The script [scripts/prepare_data.py](../scripts/prepare_data.py) loads the UltraChat dataset and saves a smaller subset to `data/processed/train` and `data/processed/validation`.

### 2. Clean conversations

The script [scripts/clean_data.py](../scripts/clean_data.py) removes malformed or low-quality chat examples. It checks:

- role ordering
- minimum message length
- minimum number of turns
- conversation length thresholds
- duplicate conversation removal

### 3. Train the adapter

The script [src/train.py](../src/train.py) performs the actual QLoRA fine-tuning.

It:

- loads the tokenizer
- creates a 4-bit NF4 quantized model
- prepares the base model for k-bit training
- adds LoRA adapters
- formats chat examples for supervised fine-tuning
- trains with `SFTTrainer`
- saves checkpoints
- optionally pushes checkpoints to Hugging Face

## Important config values

The core hyperparameters are defined at the top of [src/train.py](../src/train.py):

- `MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"`
- `MAX_SEQ_LENGTH = 1024`
- `LORA_R = 16`
- `LORA_ALPHA = 32`
- `LEARNING_RATE = 2e-4`
- `NUM_EPOCHS = 1`

## Inference workflow

Use [src/inference.py](../src/inference.py) to:

- load the base model
- load the adapter
- compare model responses to sample prompts
- evaluate conversational quality before deployment

## Recommended execution order

```bash
python scripts/prepare_data.py
python scripts/clean_data.py
python src/train.py
python src/inference.py
```

## GPU and environment notes

- Use CUDA-enabled hardware for the main training process
- Keep all outputs writable in the local project or Kaggle working directory
- If training on Kaggle, use `/kaggle/working` for checkpoints and outputs

## Troubleshooting

### CUDA not available

Run [scripts/test_gpu.py](../scripts/test_gpu.py) to confirm GPU and driver support.

### Model fails to load

Check your Python environment and make sure that the required packages from [requirements.txt](../requirements.txt) are installed.

### Dataset looks empty

Ensure both dataset preparation and cleaning scripts are run in sequence.

### Training interrupted

The training script includes checkpoint recovery logic and emergency Hub push support.
