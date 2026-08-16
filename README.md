# Qwen 3B QLoRA Fine-Tuning Project

<p align="center">
  <img src="https://img.shields.io/badge/Model-Qwen%202.5%203B-blueviolet" alt="Model" />
  <img src="https://img.shields.io/badge/Training-QLoRA-00C7B7" alt="Training" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hub-HuggingFace-FFD21E" alt="Hugging Face" />
</p>

A lightweight but production-ready fine-tuning workflow for the Qwen 2.5 3B Instruct model using QLoRA, UltraChat data cleaning, and Hugging Face Hub checkpoint publishing.

This project takes a clean, instruction-tuned conversational dataset, filters out noisy examples, and trains a compact adapter for better local generation quality without fully fine-tuning the base model.

## Why this project is useful

- Efficient fine-tuning with 4-bit quantization and low-rank adapters
- Built for GPU-heavy environments, including single-GPU setups
- Uses a cleaned conversational dataset instead of raw noisy chat dumps
- Saves intermediate checkpoints and optionally pushes them to Hugging Face
- Includes inference scripts to compare the base model and the fine-tuned adapter

## Highlights

- Base model: Qwen/Qwen2.5-3B-Instruct
- Training method: QLoRA
- Quantization: 4-bit NF4 via BitsAndBytes
- Data source: UltraChat
- Output: LoRA adapter + saved tokenizer/model artifacts
- Deployment target: local generation and Hugging Face model publishing

## Project architecture

```text
LLM/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-kaggle.txt
├── .gitignore
├── .env.example
├── data/
│   ├── processed/
│       ├── train/
│       ├── train_clean/
│       ├── validation/
│       └── validation_clean/
│  
├── model/
│   ├── adapters/
│   └── checkpoints/
├── outputs/
│   └── qwen-3b-qlora/
├── scripts/
│   ├── clean_data.py
│   ├── prepare_data.py
│   ├── test_gpu.py
│   └── test_model.py
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── inference.py
│   └── train.py
├── docs/
│   └── PROJECT_GUIDE.md
```

## How the pipeline works

1. Download and sample the UltraChat dataset
2. Filter low-quality conversations and normalize formatting
3. Train a QLoRA adapter with the Qwen 3B base model
4. Save checkpoints and final weights
5. Push completed checkpoints to the Hugging Face Hub
6. Run comparison generation with the base and fine-tuned model

## Quick start

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

```bash
python scripts/prepare_data.py
python scripts/clean_data.py
```

This creates the filtered dataset under `data/processed` for training and validation.

### 4. Train the model

```bash
python src/train.py
```

This script:

- loads Qwen/Qwen2.5-3B-Instruct
- configures 4-bit NF4 quantization
- sets up LoRA adapters
- saves model checkpoints
- pushes periodic updates to Hugging Face Hub if configured

### 5. Run inference

```bash
python src/inference.py
```

This compares the base model and the fine-tuned adapter on sample prompts.

## Training configuration

The project uses a tuned QLoRA setup in [src/train.py](src/train.py):

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Sequence length: `1024`
- LoRA rank: `16`
- LoRA alpha: `32`
- LoRA dropout: `0.05`
- Learning rate: `2e-4`
- Epochs: `1`
- Quantization: `4-bit NF4`
- Optimizer: `paged_adamw_8bit`

## Dataset cleaning logic

The cleaning pipeline in [scripts/clean_data.py](scripts/clean_data.py) filters conversations based on:

- minimum number of turns
- minimum message length
- minimum and maximum conversation length
- alternating role pattern: user → assistant → user → assistant
- duplicate conversation removal
- normalization of whitespace and content trimming

This keeps the final dataset focused on high-quality chat examples rather than noisy raw conversations.

## Hardware requirements

### Recommended

- NVIDIA GPU with CUDA support
- 12 GB+ VRAM preferred for smooth QLoRA training
- Linux or Windows with supported CUDA drivers

### Minimum

- CUDA-capable GPU
- At least 8 GB VRAM for smaller experiments
- CPU fallback is possible for testing, but training is much slower

## Hugging Face setup

Before pushing checkpoints, authenticate with Hugging Face:

```bash
huggingface-cli login
```

The training script expects a model repo such as:

```text
llm-lab-dz/qwen-3b-qlora-ultrachat
```

If you want to change the repository, edit the `HF_REPO_ID` constant inside [src/train.py](src/train.py).

## Kaggle note

If you run this project in Kaggle, remember that input folders are often read-only. Keep your outputs under `/kaggle/working` and write checkpoints there instead of to the read-only dataset folders.

## Development notes

Useful helper scripts:

- [scripts/test_gpu.py](scripts/test_gpu.py): checks PyTorch and CUDA availability
- [scripts/test_model.py](scripts/test_model.py): loads a base Qwen model and runs a quick generation test
- [src/inference.py](src/inference.py): compares base vs fine-tuned output
- [src/evaluate.py](src/evaluate.py): placeholder for evaluation logic

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Qwen team for the base model
- Hugging Face for Transformers, PEFT, TRL, and Datasets
- UltraChat for the training corpus

---

Built for efficient, modern LLM fine-tuning and clean experimentation.
