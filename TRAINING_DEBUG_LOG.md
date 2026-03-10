# Head-Only Jigsaw Training Debug Log

## Results Summary
- Head-only training was unstable in `float16` but became stable in `bfloat16`.
- The first loss computation was finite; NaNs appeared after the first optimizer update, indicating update-time numeric instability in the fp16 path.
- A full `bfloat16` run completed all `35904 / 35904` steps, but validation performance was weak and plateaued by epoch 1.
- The final crash was a post-training checkpoint reload issue (`.bin` expected, sharded `.safetensors` present), not a training stability failure.

## Scope
This document tracks the debugging history for:
- `compexp_toxicity/head_only/llama_jigsaw_toxicity_classification_only_head_train.py`
- `nautilus-jigsaw/job-run.yaml`

Goal: run head-only sequence classification on Jigsaw toxicity without LoRA/PEFT.

---

## Baseline Intended Setup
- Base model: `meta-llama/Llama-3.1-8B`
- Task: multi-label toxicity (6 labels)
- Trainable params: classifier head only (`model.score`)
- Frozen params: Llama backbone (`model.model`)
- Loss: `BCEWithLogitsLoss` (+ optional `pos_weight`)

---

## What Was Changed (Chronological)

### 1) Removed LoRA/PEFT path
- Removed:
  - `LoraConfig`
  - `prepare_model_for_kbit_training`
  - `get_peft_model`
  - PEFT-related imports/config blocks
- Kept `AutoModelForSequenceClassification` head-only strategy.

### 2) Added explicit head-only freezing
- Added:
  - freeze `model.model.parameters()`
  - train `model.score.parameters()`
- Added sanity prints:
  - trainable param count
  - trainable param names (expected: `score.weight`)

### 3) Initial no-quantized runs on 3090
- Failure: CUDA OOM on model move to GPU.
- Conclusion: full-precision load too large for 24GB VRAM in this pipeline.

### 4) Introduced memory-oriented load settings
- Added:
  - `torch_dtype=torch.float16`
  - `low_cpu_mem_usage=True`
- Result: model load proceeded past previous OOM point.

### 5) Tried Trainer AMP (`fp16=True`)
- Failure: `ValueError: Attempting to unscale FP16 gradients.`
- Action: removed `fp16=True` from `TrainingArguments`.

### 6) Tried casting classifier head to float32
- Added temporarily: `model.score = model.score.float()`
- Failure: dtype mismatch (`Half != float`) at linear layer.
- Action: removed that cast.

### 7) Persistent log capture on PVC
- In `job-run.yaml`, piped training output via:
  - `python ... | tee /workspace/logs/head_only_run_<timestamp>.log`
- Purpose: retain logs even if pod logs disappear.

### 8) Stability tuning attempt #1
- Lowered LR from `1e-3` to `1e-4`.
- Cast logits to fp32 before loss:
  - `logits = outputs.get("logits").float()`
- Added `max_grad_norm=1.0`.
- Added non-finite loss guard:
  - `if not torch.isfinite(loss): raise RuntimeError(...)`
- Observed: still unstable (`grad_norm: nan`, loss collapse to `0.0`, runtime failures).

### 9) Checked imbalance ratios
- Computed label `neg/pos` ratios (approx):
  - toxic: `9.43`
  - severe_toxic: `99.04`
  - obscene: `17.89`
  - threat: `332.83`
  - insult: `19.26`
  - identity_hate: `112.57`
- Hypothesis: very high `pos_weight` may be destabilizing fp16 training.

### 10) Stability tuning attempt #2
- Capped `pos_weight` to `30.0`.
- Added debug prints for first batch:
  - labels min/max
  - logits min/max
  - NaN/Inf checks on logits
- Observed:
  - first-batch logits finite
  - still hit `Non-finite loss detected` very early.

### 11) Isolation pass
- Set:
  - `learning_rate=1e-6`
  - `weight_decay=0.0`
  - `pos_weight=None` (disable weighting)
- Verified script on PVC reflects these values.
- Confirmed via newest run log (`head_only_run_20260305_204003.log`) that:
  - no `pos_weight` print appears,
  - but `Non-finite loss detected` still occurs very early.
- Conclusion: instability persists even with `pos_weight` disabled.

### 12) Added focused loss-path diagnostics (current)
- Updated `CustomTrainer.compute_loss`:
  - debug for first 20 calls only,
  - print labels/logits min-max and NaN/Inf checks,
  - compute BCE with `reduction="none"` and print `loss_mat` stats,
  - print scalar loss per debug step.
- Purpose:
  - identify whether non-finite values originate in labels, logits, or BCE loss elements,
  - avoid log spam by limiting debug to early calls.

### 13) Diagnostic result from focused loss-path logging
- First debug call was finite:
  - labels finite
  - logits finite
  - unreduced BCE finite
  - scalar loss finite
- Second debug call failed:
  - logits had already become `NaN`
  - unreduced BCE had `NaN`
  - scalar loss became `nan`
- Interpretation:
  - the first forward/loss computation was valid
  - the instability happened after the first optimizer update
  - this pointed away from labels and `pos_weight`, and toward update-time numeric instability in the fp16 path

### 14) Switched model dtype from `float16` to `bfloat16`
- Changed:
  - `torch_dtype=torch.float16`
  - to `torch_dtype=torch.bfloat16`
- Reason:
  - keep memory usage lower than full precision
  - use a numerically safer dtype than fp16
- Expected effect:
  - reduce update-time overflow / NaN behavior without returning to the earlier OOM regime

### 15) Stable full training run achieved
- Stable run log:
  - `/workspace/logs/head_only_run_20260307_103326.log`
- Training completed:
  - `35904 / 35904` steps
- Final eval at epoch 2.0:
  - `eval_loss = 1.4546139240264893`
  - `eval_micro_f1 = 0.07346253880359091`
  - `eval_macro_f1 = 0.06943636481719113`
  - `eval_samples_f1 = 0.04786030592209311`
  - `eval_subset_accuracy = 0.002757237749091365`
- Interpretation:
  - head-only training can be made operationally stable on available hardware
  - final task performance is poor

### 15.1) No meaningful improvement from epoch 1 to epoch 2
Epoch 1 eval:
- `eval_loss = 1.454614281654358`
- `eval_micro_f1 = 0.07346253880359091`
- `eval_macro_f1 = 0.06943636481719113`
- `eval_samples_f1 = 0.04786030592209311`
- `eval_subset_accuracy = 0.002757237749091365`

Epoch 2 eval:
- `eval_loss = 1.4546139240264893`
- `eval_micro_f1 = 0.07346253880359091`
- `eval_macro_f1 = 0.06943636481719113`
- `eval_samples_f1 = 0.04786030592209311`
- `eval_subset_accuracy = 0.002757237749091365`

Interpretation:
- metrics were effectively unchanged across epochs
- the head-only setup appears to plateau almost immediately
- this further supports treating head-only as a weak baseline rather than a promising optimization path

### 16) Post-training failure: best-checkpoint reload
- After training/eval completed, `Trainer` failed during `_load_best_model()`
- Error:
  - `FileNotFoundError: sequence_classification/checkpoint-17952/pytorch_model.bin`
- Observed checkpoint contents were sharded `.safetensors`, not `pytorch_model.bin`
- So:
  - training completed
  - evaluation completed
  - final crash happened only during best-model reload
- Immediate fix for future reruns:
  - set `load_best_model_at_end=False`

---

## Operational Issues Encountered
- Wrong `job-run.yaml` applied from a different directory once (ran `nbconvert` path, not script path).
- Existing completed wait job reused accidentally (needed delete + recreate).
- Pod/job cleanup and namespace drift caused ambiguity in logs.
- Pod logs sometimes unavailable; PVC logs are the source of truth.

---

## Key Findings

### Finding 1: head-only can be made stable
The instability was not ultimately caused by `pos_weight` alone. The decisive fix was moving from fp16 to bfloat16.

### Finding 2: head-only performance is weak
Even after stability was achieved, the validation metrics were poor:
- `eval_macro_f1 = 0.0694`
- `eval_micro_f1 = 0.0735`

This is the main modeling result from the experiment.

### Finding 3: learning plateaued almost immediately
Epoch 1 and epoch 2 evaluation metrics were effectively identical, which indicates training was stable enough to continue but did not produce meaningful improvement after the first epoch.

### Finding 4: the final run was operationally successful but not cleanly terminated
The run finished training and evaluation, but crashed during best-checkpoint reload because the expected `.bin` checkpoint file was not present.

This is a post-training artifact-management issue, not a training-stability issue.

---

## Current Recommendation

Use head-only as a baseline and pivot effort back to PEFT/LoRA.

Reason:
- the engineering question has been answered: head-only can run
- the modeling result is weak enough that further head-only tuning is unlikely to be the best use of time

If another head-only rerun is needed only for cleanup:
- keep `torch_dtype=torch.bfloat16`
- keep the conservative optimizer settings
- set `load_best_model_at_end=False`
- keep evaluation/reporting identical so results remain comparable

---

## Quick Command Checklist
```bash
# inspect newest PVC log via wait pod
kubectl exec -it <WAIT_POD> -- ls -ltr /workspace/logs | tail -n 5
kubectl exec -it <WAIT_POD> -- tail -n 200 /workspace/logs/<LATEST_LOG_FILE>
kubectl exec -it <WAIT_POD> -- grep -E "DEBUG step-call|loss_mat|Non-finite loss|Traceback|Training complete|eval_macro_f1|eval_micro_f1" /workspace/logs/<LATEST_LOG_FILE>

# inspect checkpoint layout
kubectl exec -it <WAIT_POD> -- ls -R /workspace/sequence_classification | head -300
```
