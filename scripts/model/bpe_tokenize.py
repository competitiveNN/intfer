#!/usr/bin/env python3
"""bpe_tokenize.py — encode text -> int32 token ids on stdout.

Uses the HuggingFace `tokenizers` library for proper Unicode-aware
pre-tokenization. The text can be supplied either via stdin (default) or
via --input "<text>" on the command line. The CLI uses --input to avoid
a parent/child race on the stdin pipe.

Usage: echo "Once upon a time" | python3 bpe_tokenize.py --tokenizer T.json
       python3 bpe_tokenize.py --tokenizer T.json --input "Once upon a time"
"""
import sys, struct, argparse

DEFAULT_TOKENIZER = "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/tokenizer.json"

def encode_with_tokenizers_lib(text, tok_path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    enc = tok.encode(text)
    return enc.ids

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--input", default=None,
                        help="Text to encode (default: read from stdin)")
    args = parser.parse_args()
    if args.input is not None:
        text = args.input
    else:
        text = sys.stdin.read()
    if not text: return
    try:
        ids = encode_with_tokenizers_lib(text, args.tokenizer)
    except ImportError:
        sys.stderr.write("tokenizers lib not available; install `pip install tokenizers`\n")
        return 2
    sys.stdout.buffer.write(struct.pack(f"<{len(ids)}i", *ids))

if __name__ == "__main__":
    main()
