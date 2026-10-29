"""Download OpenWebText and write the records data/text.py reads: uint16 rows of
1024 tokens (``{train,val,test}.npy``).

The documents are shuffled with seed 0, as ``datasets.Dataset.shuffle(seed=0)``
shuffles them, and the last 100k are held out. They are tokenized without
special tokens and packed in consecutive batches of 1024 documents: each
document is BOS, its tokens, EOS and 0-127 PAD tokens, the PAD counts drawn per
batch from the seed ``[0, index of the batch's first document]``, and a batch's
tokens are cut into rows with the remainder dropped. The held-out rows form 64
equal contiguous blocks: the even ones are val, the odd ones test.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from collections.abc import Callable

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

REPO = "Skylion007/openwebtext"
REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
TOKENIZER = "dvruette/openwebtext-bpe-1k"
SEQ_LEN = 1024
HELD_OUT = 100_000
PACK_DOCS = 1024
MAX_PAD = 127
BLOCKS = 64
ENCODE_BATCH = 1000
WORKERS = 16  # tokenizer processes, each holding one parquet file (up to 8 GB of RAM)

_tokenizer = None


def _load_tokenizer() -> None:
    global _tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def tokenize(path: str) -> tuple[np.ndarray, np.ndarray]:
    """One parquet file's documents: their concatenated tokens and their lengths."""
    texts = pq.read_table(path, columns=["text"]).column("text").to_pylist()
    tokens, lengths = [], []
    for start in range(0, len(texts), ENCODE_BATCH):
        batch = _tokenizer(texts[start:start + ENCODE_BATCH], add_special_tokens=False)
        for ids in batch["input_ids"]:
            tokens.append(np.asarray(ids, dtype=np.uint16))
            lengths.append(len(ids))
    return np.concatenate(tokens), np.array(lengths, dtype=np.int64)


def pack(docs: np.ndarray, lengths: np.ndarray, doc_tokens: Callable[[int], np.ndarray],
         specials: tuple[int, int, int], new_rows: Callable[[int], np.ndarray]) -> np.ndarray:
    """Pack ``docs`` (corpus indices, in order) into the rows ``new_rows(count)`` returns."""
    starts = np.arange(0, len(docs), PACK_DOCS)
    pads = np.concatenate([np.random.default_rng([0, start]).integers(
        0, MAX_PAD + 1, size=min(PACK_DOCS, len(docs) - start)) for start in starts])
    batch_rows = np.add.reduceat(lengths[docs] + 2 + pads, starts) // SEQ_LEN
    out = new_rows(int(batch_rows.sum()))
    bos, eos, pad = (np.full(1, t, dtype=np.uint16) for t in specials)
    pad_run = np.repeat(pad, MAX_PAD)
    row = 0
    for start, n in zip(starts, batch_rows, strict=True):
        pieces = []
        end = start + PACK_DOCS
        for doc, n_pad in zip(docs[start:end], pads[start:end], strict=True):
            pieces += [bos, doc_tokens(doc), eos, pad_run[:n_pad]]
        out[row:row + n] = np.concatenate(pieces)[:n * SEQ_LEN].reshape(n, SEQ_LEN)
        row += n
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write the records into")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    snapshot = snapshot_download(REPO, repo_type="dataset", revision=REVISION,
                                 allow_patterns="plain_text/*.parquet")
    folder = os.path.join(snapshot, "plain_text")
    with mp.Pool(WORKERS, initializer=_load_tokenizer) as pool:
        files = pool.map(tokenize, [os.path.join(folder, f) for f in sorted(os.listdir(folder))],
                         chunksize=1)
    lengths = np.concatenate([n for _, n in files])
    file_of = np.repeat(np.arange(len(files)), [len(n) for _, n in files])
    offset = np.concatenate([np.cumsum(n) - n for _, n in files])

    def doc_tokens(doc: int) -> np.ndarray:
        return files[file_of[doc]][0][offset[doc]:offset[doc] + lengths[doc]]

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    specials = (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)
    order = np.random.default_rng(0).permutation(len(lengths))

    # The training rows go straight to disk.
    train_path = os.path.join(args.out, "train.npy")
    train = pack(order[:-HELD_OUT], lengths, doc_tokens, specials,
                 lambda n: np.lib.format.open_memmap(train_path, mode="w+", dtype=np.uint16,
                                                     shape=(n, SEQ_LEN)))
    train.flush()
    print(f"train: {len(train)} records -> {args.out}", flush=True)

    held_out = pack(order[-HELD_OUT:], lengths, doc_tokens, specials,
                    lambda n: np.empty((n, SEQ_LEN), dtype=np.uint16))
    block = np.arange(len(held_out)) // -(-len(held_out) // BLOCKS)
    for split, parity in (("val", 0), ("test", 1)):
        rows = held_out[block % 2 == parity]
        np.save(os.path.join(args.out, f"{split}.npy"), rows)
        print(f"{split}: {len(rows)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
