# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanochat is a full-stack LLM training framework designed to create ChatGPT-like models on a budget ($100-$1000). It's a Python-based machine learning project that implements the complete pipeline: tokenization, pretraining, fine-tuning (SFT), reinforcement learning, evaluation, and web serving.

## Key Architecture

### Pipeline Stages
1. **Tokenizer Training** (`rustbpe/` + `scripts/tok_train.py`): Custom Rust BPE tokenizer
2. **Base Model Pretraining** (`scripts/base_train.py`): Transformer pretraining on web data
3. **Midtraining** (`scripts/mid_train.py`): Teaches conversation tokens, tool use, multiple choice
4. **Supervised Fine-Tuning** (`scripts/chat_sft.py`): Domain adaptation to conversations
5. **Reinforcement Learning** (`scripts/chat_rl.py`): Optional, currently only on GSM8K
6. **Evaluation** (`scripts/*_eval.py` + `tasks/`): Multiple benchmark tasks
7. **Deployment** (`scripts/chat_web.py` + `nanochat/ui.html`): Web UI for ChatGPT-like interaction

### Core Components
- **`nanochat/gpt.py`**: The main Transformer neural network implementation
- **`nanochat/engine.py`**: Efficient inference with KV Cache
- **`nanochat/dataloader.py`**: Tokenizing distributed data loader
- **`nanochat/adamw.py`**: Distributed AdamW optimizer (for embeddings)
- **`nanochat/muon.py`**: Distributed Muon optimizer (for matrix parameters)
- **`nanochat/execution.py`**: Allows LLM to execute Python code as tool

### Evaluation Tasks (`tasks/`)
- `arc.py`: Multiple choice science questions
- `gsm8k.py`: Grade School Math questions
- `humaneval.py`: Simple Python coding task
- `mmlu.py`: Multiple choice questions, broad topics
- `spellingbee.py`: Task teaching model to spell/count letters
- `smoltalk.py`: Conglomerate dataset of SmolTalk from HuggingFace

## Common Development Tasks

### Environment Setup
```bash
# Install uv package manager if not present
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
uv sync --extra gpu  # or --extra cpu for CPU-only
source .venv/bin/activate

# Build Rust tokenizer
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
```

### Training Pipelines
```bash
# Full $100 training pipeline (4 hours on 8XH100)
bash speedrun.sh

# $800 training pipeline (d32 model, ~33 hours)
bash run1000.sh

# CPU/MPS example (smaller models)
bash dev/runcpu.sh
```

### Individual Stages
```bash
# Tokenizer training
python -m scripts.tok_train --max_chars=2000000000

# Base model pretraining (multi-GPU)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20

# Midtraining
torchrun --standalone --nproc_per_node=8 -m scripts.mid_train

# Supervised Fine-Tuning
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft

# Reinforcement Learning (optional)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl
```

### Evaluation
```bash
# Tokenizer evaluation
python -m scripts.tok_eval

# Base model evaluation (CORE score, bits per byte)
torchrun --standalone --nproc_per_node=8 -m scripts.base_loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval

# Chat model evaluation (all tasks)
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i mid  # or -i sft, -i rl

# Specific task evaluation
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft -a GSM8K
```

### Inference and Serving
```bash
# CLI chat interface
python -m scripts.chat_cli -p "Why is the sky blue?"

# Web UI (ChatGPT-like interface)
python -m scripts.chat_web  # Serves on http://localhost:8000
```

### Testing
```bash
# Run tokenizer tests
python -m pytest tests/test_rustbpe.py -v -s

# Run engine tests
python -m pytest tests/test_engine.py -v -s

# Skip slow tests
python -m pytest -m "not slow"
```

## Configuration and Customization

### Model Size Scaling
To train larger models (e.g., d26 for ~$300 tier):
1. Download more data shards: `python -m nanochat.dataset -n 450`
2. Adjust `--depth` parameter in training scripts
3. Reduce `--device_batch_size` if OOM (32 → 16 → 8 → etc.)

### Data Management
- Pretraining data shards: ~250M characters each, stored in `~/.cache/nanochat/`
- Download shards: `python -m nanochat.dataset -n <num_shards>`
- Total available shards: 1822

### Custom Identity/Personality
See `dev/gen_synthetic_data.py` for example synthetic data generation. Custom conversations can be mixed into midtraining and SFT stages via `--identity_data_path`.

## Key Design Principles

1. **Minimalism**: Single cohesive codebase, no giant configuration objects
2. **Accessibility**: Designed for $100-$1000 training budgets
3. **End-to-end**: Complete pipeline from raw data to web UI
4. **Hackability**: Clean, readable code meant to be forked and modified
5. **Modern Architecture**: Rotary embeddings, QK norm, GQA support, no biases

## Important Notes

- Uses `uv` as package manager (modern alternative to pip)
- Rust tokenizer requires Cargo/Rust installation
- Multi-GPU training uses `torchrun` with `--standalone` flag
- Intermediate artifacts stored in `~/.cache/nanochat/`
- Final report generated as `report.md` with evaluation metrics
- WandB logging optional (set `WANDB_RUN` environment variable)

## File Structure Reference

```
nanochat/           # Core Python package
├── gpt.py          # Transformer model
├── engine.py       # KV cache inference
├── dataloader.py   # Distributed data loader
├── adamw.py        # AdamW optimizer
├── muon.py         # Muon optimizer
└── ui.html         # Web UI frontend

scripts/            # Training/evaluation scripts
├── base_train.py   # Pretraining
├── mid_train.py    # Midtraining
├── chat_sft.py     # Supervised Fine-Tuning
├── chat_rl.py      # Reinforcement Learning
└── *_eval.py       # Evaluation scripts

tasks/              # Evaluation benchmarks
rustbpe/            # Rust BPE tokenizer
dev/                # Development utilities
tests/              # Test files
```