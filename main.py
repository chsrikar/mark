#!/usr/bin/env python3
"""
main.py - Unified command-line interface for the Mark LLM project.

Available commands:
  tokenize   - Train a byte-level BPE tokenizer on text + code
  prepare    - Prepare train.bin / val.bin from raw datasets
  train      - Train the GPT model (presets: tiny, 30m)
  generate   - Generate text from a trained checkpoint
  app        - Start the web studio (Tokenizer Lab & Model Playground)

Examples:
  python main.py tokenize --input ML-Arxiv-Papers.csv
  python main.py prepare --code-dir code_corpus
  python main.py train --preset tiny
  python main.py generate --prompt "Attention is all you need"
  python main.py app --port 8000
"""
import sys


def print_help():
    print("""Mark - Personal LLM Toolkit

Usage:
  python main.py <command> [options]

Commands:
  tokenize   Train a byte-level BPE tokenizer (train_tokenizer.py)
  prepare    Tokenize raw text & code into binary arrays (prepare_data.py)
  train      Train the GPT model with checkpoints (train.py)
  generate   Generate text from trained weights (generate.py)
  app        Launch the local web studio (tokenizer_app.py)

Run 'python main.py <command> --help' for options specific to a command.
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    sub_args = sys.argv[2:]

    if cmd == "tokenize":
        import train_tokenizer
        sys.argv = [sys.argv[0] + " tokenize"] + sub_args
        train_tokenizer.main()

    elif cmd == "prepare":
        import prepare_data
        prepare_data.main(sub_args)

    elif cmd == "train":
        import train
        train.main(sub_args)

    elif cmd == "generate":
        import generate
        generate.main(sub_args)

    elif cmd in ("app", "studio", "web"):
        import tokenizer_app
        sys.argv = [sys.argv[0] + " app"] + sub_args
        tokenizer_app.main()

    else:
        print(f"Unknown command: '{cmd}'\n")
        print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
