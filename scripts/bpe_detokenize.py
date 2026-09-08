#!/usr/bin/env python3
"""bpe_detokenize.py — read one int32 token id (or --id N), write its bytes.

Qwen3.5 (and most modern tokenizers using HuggingFace `tokenizers`) use a
ByteLevel pre-tokenizer and decoder (GPT-2 byte-to-unicode mapping). The
vocab stores tokens as strings of printable characters; to recover the
original byte string we reverse the mapping.

Two modes:
  * Default: read 4 bytes of LE int32 from stdin, decode, write bytes to stdout.
  * With --id N: use N directly (avoids the parent/child popen race).
"""
import sys, json, struct, argparse

DEFAULT_TOKENIZER = "/var/home/fra/data/models/safetensors/Qwen3.5-0.8B/tokenizer.json"

def _make_encoder():
    bs = list(range(ord("!"), ord("~")+1)) \
       + list(range(ord("¡"), ord("¬")+1)) \
       + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))

def _make_decoder():
    enc = _make_encoder()
    return {v: k for k, v in enc.items()}

_DECODER = None
def _get_decoder():
    global _DECODER
    if _DECODER is None:
        _DECODER = _make_decoder()
    return _DECODER

_INV_VOCAB = None
def _get_inv_vocab(path):
    global _INV_VOCAB
    if _INV_VOCAB is None:
        with open(path, "r", encoding="utf-8") as f:
            tok = json.load(f)
        vocab = tok["model"]["vocab"]
        _INV_VOCAB = {idx: s for s, idx in vocab.items()}
    return _INV_VOCAB

def detok_one(tid: int, inv_vocab, u2b) -> bytes:
    s = inv_vocab.get(tid)
    if s is None:
        return b""
    out = bytearray()
    for ch in s:
        if ch in u2b:
            out.append(u2b[ch])
        else:
            out += ch.encode("utf-8")
    return bytes(out)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--id", type=int, default=None)
    args = parser.parse_args()
    if args.id is not None:
        tid = args.id
    else:
        data = sys.stdin.buffer.read()
        if not data: return
        (tid,) = struct.unpack("<i", data[:4])
    inv = _get_inv_vocab(args.tokenizer)
    u2b = _get_decoder()
    sys.stdout.buffer.write(detok_one(tid, inv, u2b))

if __name__ == "__main__":
    main()
