"""Quality of text sample sets: generative perplexity under a judge LM, the pooled
unigram entropy of the judge's tokens, and MAUVE against held-out records.

The ``samples.json`` files of ``binary_diffusion.sample`` must come from one corpus
and hold as many texts each. They share the references: per reference seed, that
many distinct records of ``split``, decoded like the samples. MAUVE is averaged
over the reference and clustering seeds.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter

import hydra
import mauve
import numpy as np
import torch
from mauve.utils import featurize_tokens_from_model
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..data.factory import create_tokenizer
from ..data.text import reference_texts


def token_windows(ids: list[int], prefix: int, context_length: int,
                  stride: int) -> list[tuple[list[int], int, int]]:
    """Sliding windows over ``[prefix, *ids]`` that score every text token once:
    each window's ids and the half-open range of its target positions."""
    tokens = [prefix, *ids]
    windows = []
    end = min(context_length, len(tokens))
    if end > 1:
        windows.append((tokens[:end], 1, end))
    while end < len(tokens):
        start = end
        end = min(end + stride, len(tokens))
        begin = max(0, end - context_length)
        windows.append((tokens[begin:end], start - begin, end - begin))
    return windows


def histogram_entropy(counts: Counter) -> float:
    n = counts.total()
    return -sum((v / n) * math.log(v / n) for v in counts.values()) if n else 0.0


@torch.no_grad()
def score(texts: list[str], model, tokenizer, context_length: int, batch_size: int,
          stride: int) -> tuple[float, Counter]:
    """The judge's total NLL of the texts, and the counts of their judge tokens. The
    unscored BOS before a text lets its first token be scored too."""
    prefix = tokenizer.bos_token_id
    counts, windows = Counter(), []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        counts.update(ids)
        windows += token_windows(ids, prefix, context_length, stride)
    nll = 0.0
    for offset in range(0, len(windows), batch_size):
        chunk = windows[offset:offset + batch_size]
        width = max(len(values) for values, _, _ in chunk)
        ids = torch.full((len(chunk), width), prefix, dtype=torch.long, device=model.device)
        attention = torch.zeros_like(ids)
        for row, (values, _, _) in enumerate(chunk):
            ids[row, :len(values)] = torch.tensor(values, device=model.device)
            attention[row, :len(values)] = 1
        logits = model(ids, attention_mask=attention, use_cache=False).logits
        for row, (_, start, end) in enumerate(chunk):
            z = logits[row, start - 1:end - 1].float()
            target = ids[row, start:end]
            losses = torch.logsumexp(z, -1) - z.gather(-1, target[:, None]).squeeze(-1)
            nll += float(losses.double().sum())
    return nll, counts


def featurize(texts: list[str], model, tokenizer, max_text_length: int,
              batch_size: int) -> np.ndarray:
    """MAUVE features: the judge's last hidden state at the final token of each
    text's first ``max_text_length`` tokens. An empty text is the context token
    alone, so it stays in the distribution."""
    rows = [tokenizer.encode(text, add_special_tokens=False, truncation=False)[:max_text_length]
            or [tokenizer.bos_token_id] for text in texts]
    return featurize_tokens_from_model(
        model, [torch.tensor([ids], dtype=torch.long) for ids in rows], batch_size
    ).float().numpy()


@hydra.main(config_path="../../../configs/eval", config_name="text_quality", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runs = []
    for path in cfg.samples:
        with open(path) as f:
            runs.append(json.load(f))
    data, n = runs[0]["run_config"]["data"], len(runs[0]["texts"])
    if any(run["run_config"]["data"] != data or len(run["texts"]) != n for run in runs):
        raise ValueError("the samples files must come from one corpus and hold as many texts")
    run_cfg = OmegaConf.create(runs[0]["run_config"])
    corpus_tokenizer = create_tokenizer(run_cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.judge, revision=cfg.judge_revision)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.judge, revision=cfg.judge_revision, dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_implementation,
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    m = cfg.mauve
    references = [featurize(reference_texts(run_cfg, corpus_tokenizer, cfg.split, n, seed),
                            model, tokenizer, m.max_text_length, cfg.batch_size)
                  for seed in m.reference_seeds]

    results = []
    for path, run in zip(cfg.samples, runs, strict=True):
        nll, counts = score(run["texts"], model, tokenizer, cfg.context_length,
                            cfg.batch_size, cfg.stride)
        q = featurize(run["texts"], model, tokenizer, m.max_text_length, cfg.batch_size)
        scores = [mauve.compute_mauve(p_features=p, q_features=q, num_buckets=m.num_buckets,
                                      seed=seed, use_float64=m.float64).mauve
                  for p in references for seed in m.clustering_seeds]
        results.append({"samples": path, "genppl": math.exp(nll / counts.total()),
                        "entropy": histogram_entropy(counts), "mauve": float(np.mean(scores))})
        print(json.dumps(results[-1]), flush=True)
    with open(os.path.join(cfg.out, "result.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
