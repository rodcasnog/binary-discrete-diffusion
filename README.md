# Binary Discrete Diffusion

Code for [Binary Discrete Diffusion: Candidate-Set Diffusion for Logic and Planning Tasks](https://arxiv.org/abs/2610.28543). The noisy state of binary diffusion is a set of candidate tokens per position, a binary membership vector. This repository trains the masked, uniform, binary symmetric and binary absorbing kernels with a cavity head and an adaptive time sampler, and samples them with the confidence-guided Psi-sampler, on Sudoku, Boxoban, graph coloring, Text8, TinyStories and OpenWebText.

## Install

```bash
uv sync
```

The paper's runs set `XLA_FLAGS="--xla_gpu_triton_gemm_any=true --xla_gpu_enable_latency_hiding_scheduler=true"`, and large batches may need `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.

## Data

Each script writes its task's arrays to the directory the configs read (`data.dataset_path`):

| Task | Command | Source |
|---|---|---|
| Sudoku | `uv run python scripts/prep_sudoku.py --out data/sudoku` | `sapientinc/sudoku-extreme`, val/test split `rodcasnog/sudoku-extreme-split` |
| Boxoban | `uv run python scripts/prep_boxoban.py --out data/boxoban` | `google-deepmind/boxoban-levels`, `AlignmentResearch/boxoban-astar-solutions` |
| Graph coloring | none, generated on the fly | |
| Text8 | `uv run python scripts/prep_text8.py --out data/text8` | `afmck/text8`, and the character tokenizer |
| TinyStories | `uv run python scripts/prep_tinystories.py --out data/tinystories` | `roneneldan/TinyStories`, tokenizer `rodcasnog/tinystories-bpe-1k` |
| OpenWebText | `uv run python scripts/prep_owt.py --out data/owt` | `Skylion007/openwebtext`, tokenizer `dvruette/openwebtext-bpe-1k` |

## Training

`configs/paper/<task>/<kernel>.yaml` is the recipe of one paper model (tasks `sudoku`, `boxoban`, `graphcolor`,
`text8`, `tinystories`, `owt`; kernels `masked`, `uniform`, `binary_symmetric`, `binary_absorbing`):

```bash
uv run python -m binary_diffusion.train --config-name paper/sudoku/binary_symmetric training.seed=0 \
    hydra.run.dir=outputs/sudoku-bsym
```

The run directory holds `checkpoints/` (to resume), `checkpoints_ema/` (EMA parameters, for evaluation) and `metrics.jsonl`; rerunning the same command resumes the run. `wandb.mode=online` also logs to Weights & Biases. `training.global_batch_size` is split across the GPUs of one host, without gradient accumulation.

## Sampling and evaluation

`sample`, `solve` and `nelbo` load `checkpoint_dir=<run>/checkpoints_ema/<step>` with the run's `.hydra/config.yaml`, and every entry point writes into `out`. The sampler settings are `sampler.*` (`configs/eval/sampler/psi.yaml`); the defaults are ancestral sampling, and the paper's appendix lists each model's selected settings.

```bash
# Solve rate of a logic model; hard=true selects the hard instances, split=val n=2048 a validation subset.
uv run python -m binary_diffusion.eval.solve checkpoint_dir=outputs/sudoku-bsym/checkpoints_ema/1000000 \
    sampler.nfe=256 out=outputs/sudoku-bsym/solve
# NELBO in nats per token, and bits per character for text.
uv run python -m binary_diffusion.eval.nelbo checkpoint_dir=outputs/sudoku-bsym/checkpoints_ema/1000000 \
    out=outputs/sudoku-bsym/nelbo
```

## Pretrained text models

The twelve text models (Text8, TinyStories and OpenWebText, four kernels each) are run directories in [rodcasnog/binary-discrete-diffusion](https://huggingface.co/rodcasnog/binary-discrete-diffusion). Text8 models read the tokenizer that `prep_text8.py` writes, TinyStories samples take their lengths from `data/tinystories`, and NELBO and MAUVE read the corpus.

```bash
uv run hf download rodcasnog/binary-discrete-diffusion --include "text8/binary_symmetric/*" --local-dir outputs/hf
# Text samples (samples.json) with the paper's selected sampler for this model, then their generative
# perplexity, unigram entropy and MAUVE.
uv run python -m binary_diffusion.sample checkpoint_dir=outputs/hf/text8/binary_symmetric/checkpoints_ema/1000000 \
    sampler.nfe=256 sampler.spacing=log sampler.eta=1 sampler.lam=1 out=outputs/text8-bsym/samples
uv run python -m binary_diffusion.eval.text_quality 'samples=[outputs/text8-bsym/samples/samples.json]' \
    out=outputs/text8-bsym/quality
```

## Citation

```bibtex
@misc{casadonoguerales2026binarydiffusion,
  title         = {Binary Discrete Diffusion: Candidate-Set Diffusion for
                   Logic and Planning Tasks},
  author        = {Casado Noguerales, Rodrigo and von R{\"u}tte, Dimitri
                   and Sch{\"o}lkopf, Bernhard and Hofmann, Thomas
                   and Raoufi, Aran},
  year          = {2026},
  eprint        = {2610.28543},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2610.28543},
}
```

## License

Apache-2.0, see [LICENSE](LICENSE).
