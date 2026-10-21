"""
mark - A personal LLM built from scratch.

Pipeline:
  1. Tokenizer training (train_tokenizer.py)
  2. Data preparation (prepare_data.py)
  3. Model architecture (model.py)
  4. Training loop (train.py)
  5. Text generation (generate.py)
  6. Interactive Web Studio (tokenizer_app.py)
"""

from model import GPT, GPTConfig, RMSNorm, Attention, SwiGLU, Block
from generate import generate_text, load_model, resolve_tokenizer

__all__ = [
    "GPT",
    "GPTConfig",
    "RMSNorm",
    "Attention",
    "SwiGLU",
    "Block",
    "generate_text",
    "load_model",
    "resolve_tokenizer",
]
