# Jigsaw Toxicity CoMExp

Multi-label toxicity fine-tuning and compositional explanation workflow.

## Folders

- `compexp_toxicity/`: notebook and Python training scripts
- `nautilus-jigsaw/`: Kubernetes manifests for Nautilus jobs and PVC

## Quick Start

1. Configure secrets (`HF_TOKEN`, and optionally `KAGGLE_USERNAME`, `KAGGLE_KEY`).
2. Run smoke test locally:

```bash
python compexp_toxicity/llama_jigsaw_toxicity_classification_train.py --smoke-test
```

3. Run on Nautilus with manifests in `nautilus-jigsaw/`.
