"""Download ``afmck/text8`` and write the records data/text.py reads: the
canonical train, validation and test character streams cut into rows of 256
characters (``{train,val,test}.npy``, uint16; a final partial row is dropped),
and the character tokenizer in ``<out>/tokenizer``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

REPO = "afmck/text8"
REVISION = "58c74e966ccc66eab2de6b52fad5b14ddb259fa4"
ALPHABET = " abcdefghijklmnopqrstuvwxyz"     # token ids 0..26
SEQ_LEN = 256
SPLITS = {"train": "train", "val": "validation", "test": "test"}


def char_tokenizer() -> PreTrainedTokenizerFast:
    """One token per character, with ids in alphabet order."""
    tokenizer = Tokenizer(models.WordLevel(
        {c: i for i, c in enumerate(ALPHABET)}, unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the records into")
    out = ap.parse_args().out
    os.makedirs(out, exist_ok=True)

    char_tokenizer().save_pretrained(os.path.join(out, "tokenizer"))
    # The tokenizer's ids as a byte lookup (the pinned corpus holds only the alphabet).
    lookup = np.zeros(256, dtype=np.uint16)
    lookup[np.frombuffer(ALPHABET.encode(), dtype=np.uint8)] = np.arange(len(ALPHABET))
    for split, name in SPLITS.items():
        path = hf_hub_download(REPO, f"data/{name}-00000-of-00001.parquet",
                               repo_type="dataset", revision=REVISION)
        text = pq.read_table(path).column("text")[0].as_py()
        ids = lookup[np.frombuffer(text.encode(), dtype=np.uint8)]
        rows = len(ids) // SEQ_LEN
        np.save(os.path.join(out, f"{split}.npy"), ids[:rows * SEQ_LEN].reshape(rows, SEQ_LEN))
        print(f"{split}: {rows} records -> {out}", flush=True)


if __name__ == "__main__":
    main()
