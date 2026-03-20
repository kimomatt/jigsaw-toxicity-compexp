# Jigsaw Toxicity Compexp

Jigsaw toxicity classification experiments plus a dataset-agnostic concept annotation library for compositional explanations.

## Repo Layout

- `concepts/`: concept extraction package and spec for Tier 1 lexical concepts, Tier 2 linguistic concepts, and Tier 3 cluster placeholders
- `demo_concepts.py`: local concept demo and Jigsaw CSV integration checks
- `compexp_toxicity/peft/`: QLoRA / PEFT multi-label training workflow
- `compexp_toxicity/head_only/`: head-only multi-label training and export workflow
- `nautilus-jigsaw/`: Nautilus job manifests and PVC config

## Prerequisites

- Python 3.10+
- `HF_TOKEN` for gated Hugging Face models such as `meta-llama/Llama-3.1-8B`
- `KAGGLE_USERNAME` and `KAGGLE_KEY` if you want the training scripts to auto-download the Jigsaw competition data
- For concept extraction: `spacy` plus the English model `en_core_web_sm`

Install the lightweight concept dependencies with:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The training scripts require additional ML dependencies beyond `requirements.txt`. The Nautilus manifests install those explicitly inside the job container.

## Quick Start

Run the concept demo locally:

```bash
python demo_concepts.py
```

Run a PEFT smoke test locally:

```bash
python compexp_toxicity/peft/llama_jigsaw_toxicity_classification_train.py --smoke-test
```

Run a head-only smoke test locally:

```bash
python compexp_toxicity/head_only/llama_jigsaw_toxicity_classification_only_head_train.py --smoke-test
```

## Training Workflows

### PEFT / QLoRA

Main script:

```bash
python compexp_toxicity/peft/llama_jigsaw_toxicity_classification_train.py
```

This path trains a multi-label sequence classifier with 4-bit quantization, LoRA adapters, stratified train/validation splitting, and optional threshold sweeping.

### Head-Only

Main script:

```bash
python compexp_toxicity/head_only/llama_jigsaw_toxicity_classification_only_head_train.py
```

This path freezes the backbone, trains only the classification head, and can also export predictions from a saved model:

```bash
python compexp_toxicity/head_only/llama_jigsaw_toxicity_classification_only_head_train.py \
  --export-only \
  --saved-model-dir sequence_classification/saved_model
```

## Nautilus Manifests

- `nautilus-jigsaw/job-run.yaml`: older single-job runner template
- `nautilus-jigsaw/job-run-peft.yaml`: PEFT training job
- `nautilus-jigsaw/job-export-head-only.yaml`: head-only export job
- `nautilus-jigsaw/job-wait.yaml`: utility manifest for waiting / debugging
- `nautilus-jigsaw/pvc.yaml`: persistent volume claim

These manifests expect workspace files under `/workspace` and secrets for Hugging Face and Kaggle access.

## Concepts

The concept system is documented in `concepts/SPEC.md`. Current implementation highlights:

- Tier 1: top-k word presence concepts fit from train text
- Tier 2: spaCy-based explicit linguistic concepts
- Tier 2 baseline: archived rule primitives kept for comparison
- Tier 3: cluster concept scaffolding

The primary package entry points are:

- `concepts.build.build_concept_set`
- `concepts.build_tier1_vocabulary`

## Notes

- Training scripts auto-download and unzip the Kaggle Jigsaw dataset if `train.csv` is not already present under `jigsaw-toxic-comment-classification-challenge/`.
- Default outputs are written under `sequence_classification/`.
