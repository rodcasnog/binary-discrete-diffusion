"""Download TinyStories (V2-GPT4) and write the records data/text.py reads: one
story per uint16 record of 512 tokens, the story, its EOS and PAD up to the end
(``{train,val,test}.npy``); stories longer than a record are dropped. Every story
is first folded onto printable ASCII (``normalise``), whose result depends on
the Python version's ``unicodedata``. The held-out stories alternate between val
(even records) and test (odd ones).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import unicodedata

import numpy as np
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TOKENIZER = "rodcasnog/tinystories-bpe-1k"
EOS = "<|endoftext|>"
SEQ_LEN = 512
ENCODE_BATCH = 2000

# Above this share of CJK a story is an English/Chinese translation pair rather
# than English with a stray glyph.
CJK_SHARE = 0.02

# Corpus characters with a plain-ASCII equivalent. Applied before NFKC, which
# would otherwise decompose U+00B4 into space + combining acute and hide it
# here, turning "it´s" into "it s". U+2764 is the one pictograph the corpus
# uses as a word ("I<3 U"); the other four are decoration and simply go.
CHAR_MAP = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x00B4: "'", 0x0092: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
    0x00AB: '"', 0x00BB: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-",
    0x2015: "-", 0x2212: "-", 0x2500: "-",
    0x2026: "...", 0x00B7: ".", 0x2022: "*",
    0x00DF: "ss", 0x0153: "oe", 0x00E6: "ae",
    0x2764: " love ",
    # No in-place word form: a character map cannot reorder a sentence, so
    # mapping U+20AC to " euros" would render "EUR5" as "euros5".
    0x00AD: "", 0x20AC: "", 0x00A3: "", 0x00A2: "", 0x2122: "", 0x00A7: "",
    0x00A1: "", 0x00BF: "", 0x00B0: "", 0xFE0F: "",
}
ASCII_OK = frozenset(chr(c) for c in range(32, 127)) | {"\n"}
SPACE_RUN = re.compile(r"[^\S\n]+")
BLANK_LINES = re.compile(r"\n[^\S\n]*(?:\n[^\S\n]*)+")
CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")


def normalise(doc: str) -> str:
    """Fold one story onto printable ASCII, or return "" to drop it.

    NFKD puts accents on their own so the filter keeps the base letter -- cafe,
    pinata, not caf, pi ata. Newlines are paragraph and dialogue boundaries, so
    only runs of blank lines collapse.
    """
    if len(CJK.findall(doc)) > CJK_SHARE * max(len(doc), 1):
        return ""
    doc = doc.translate(CHAR_MAP)
    doc = unicodedata.normalize("NFKC", doc).translate(CHAR_MAP)
    doc = "".join(c for c in unicodedata.normalize("NFKD", doc) if c in ASCII_OK)
    return BLANK_LINES.sub("\n", SPACE_RUN.sub(" ", doc)).strip()


def stories(name: str) -> list[str]:
    """The normalised stories of one upstream file, in order."""
    path = hf_hub_download(REPO, f"TinyStoriesV2-GPT4-{name}.txt", repo_type="dataset",
                           revision=REVISION)
    with open(path, encoding="utf-8") as f:
        raw = [d for d in (x.strip("\n") for x in f.read().split(EOS)) if d]
    with mp.Pool() as pool:
        return [doc for doc in pool.imap(normalise, raw, chunksize=2000) if doc]


def records(docs: list[str], tokenizer) -> np.ndarray:
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    rows = []
    for start in range(0, len(docs), ENCODE_BATCH):
        batch = tokenizer(docs[start:start + ENCODE_BATCH], add_special_tokens=False)
        for ids in batch["input_ids"]:
            if len(ids) + 1 > SEQ_LEN:
                continue
            row = np.full(SEQ_LEN, pad, dtype=np.uint16)
            row[:len(ids)] = ids
            row[len(ids)] = eos
            rows.append(row)
    return np.stack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the records into")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    held_out = records(stories("valid"), tokenizer)
    splits = {"train": records(stories("train"), tokenizer),
              "val": held_out[0::2], "test": held_out[1::2]}
    for split, rows in splits.items():
        np.save(os.path.join(args.out, f"{split}.npy"), rows)
        print(f"{split}: {len(rows)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
