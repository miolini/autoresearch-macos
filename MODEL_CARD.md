---
language:
- en
license: mit
tags:
- financial
- sec-filings
- 10-K
- small-language-model
- slm
- gpt
- domain-specific
metrics:
- bits-per-byte
pipeline_tag: text-generation
model-index:
- name: 10k-financial-slm
  results:
  - task:
      type: text-generation
    metrics:
    - name: val_bpb (financial text)
      type: bits-per-byte
      value: 1.645
    - name: val_bpb (general text baseline)
      type: bits-per-byte
      value: 2.146
---

# 10-K Financial SLM (11.5M params)

A tiny GPT language model trained exclusively on SEC 10-K filings from financial companies. ~20 experiments at 5 minutes each (~2 hours total GPU time) on a MacBook Air using Apple Silicon (MPS).

## Model Details

| Property | Value |
|----------|-------|
| Parameters | 11.5M |
| Architecture | GPT (decoder-only transformer) |
| Layers | 4 |
| Hidden dim | 256 |
| Attention heads | 2 |
| Context length | 2,048 tokens |
| Vocab size | 8,192 (BPE) |
| Training data | 1,131 SEC 10-K filings (financial companies, SIC 6000-6411) |
| Training time | ~2 hours total (~20 x 5-min experiments on Apple M-series MPS) |

## Performance

### Compression Quality (bits-per-byte)

| Model | val_bpb | Domain |
|-------|---------|--------|
| **This model (specialized)** | **1.645** | Financial 10-K text |
| Same architecture (general) | 2.146 | General web text (ClimbMix) |

**23.3% better compression** on financial text compared to the same architecture trained on general data.

### Inference Speed (MacBook Air, MPS)

| Metric | Value |
|--------|-------|
| Single sequence latency | 27ms (2,048 tokens) |
| Batched throughput | 75,000+ tokens/sec |
| Time per 10-K filing | ~1 second |
| Full SEC EDGAR database | ~22 hours |

### Cost Comparison (processing 80K filings)

| Approach | Cost |
|----------|------|
| GPT-4o API ($2.50/1M tokens) | ~$15,000 |
| Claude Sonnet 4.6 API ($3.00/1M tokens) | ~$18,000 |
| Claude Haiku 4.5 API ($1.00/1M tokens) | ~$6,000 |
| GPT-4o-mini API ($0.15/1M tokens) | ~$900 |
| **This model (local)** | **$0** |

*Prices as of March 2026. Input tokens only.*

## Training Details

Built using [Karpathy's autoresearch](https://github.com/miolini/autoresearch-macos) framework, which enables autonomous hyperparameter experimentation. An AI agent (Claude) iteratively modified the training configuration, ran 5-minute training sessions, and kept improvements.

### Key hyperparameters (after optimization)

- Learning rates: 1.5x default (Embedding: 0.9, Matrix/Muon: 0.06)
- Warmdown ratio: 0.05 (LR stays at peak for 95% of training)
- Optimizer: MuonAdamW (Muon for matrix params, AdamW for embeddings)
- Batch size: 65,536 tokens per step

### Data pipeline

1. Downloaded 10-K filing index from SEC EDGAR (2015-2025)
2. Filtered to financial companies (SIC codes 6000-6411): banks, insurance, investment firms
3. Sampled 1,500 filings, downloaded full text from EDGAR
4. Cleaned HTML/XBRL markup, removed filings that were too short or too numeric
5. Chunked into 2,048-token sequences, split 90/10 train/val
6. Trained a BPE tokenizer (8,192 vocab) on the financial text

## Intended Use

This model is a research artifact demonstrating domain-specific SLM training. Potential applications:

- **Document embeddings**: Fast similarity search over financial filings
- **Anomaly detection**: Flag filings with unusual language patterns
- **Pre-filtering**: Cheap triage before sending documents to expensive API models
- **Privacy-preserving analysis**: All processing stays on-device
- **Foundation for fine-tuning**: Starting point for downstream financial NLP tasks

## Limitations

- **Not a chatbot**: This is a base language model. It predicts next tokens, it doesn't answer questions.
- **Tiny model**: 11.5M parameters means limited capacity. It captures patterns and statistics of financial language, not deep reasoning.
- **Narrow training data**: Only financial company 10-K filings. Performance on other financial documents (earnings calls, proxy statements) is untested.
- **No safety training**: No RLHF, no content filtering. Not suitable for user-facing generation.

## How to Use

```python
import torch
from train import GPT, GPTConfig

# Load checkpoint
ckpt = torch.load("model.pt", map_location="cpu")
config = GPTConfig(**ckpt["config"])

model = GPT(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Run inference
tokens = torch.tensor([[1, 2, 3, ...]])  # your tokenized input
with torch.no_grad():
    logits = model(tokens)
```

## Citation

If you use this model in your work, please cite:

```
@misc{10k-financial-slm-2026,
  title={10-K Financial SLM: A Domain-Specific Small Language Model for SEC Filings},
  year={2026},
  url={https://github.com/harryschaefer93/autoresearch-10k-macos}
}
```

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) / [autoresearch-macos](https://github.com/miolini/autoresearch-macos) for the training framework
- [Claude Code](https://claude.ai/claude-code) for autonomous experiment orchestration
- SEC EDGAR for the public filing data
