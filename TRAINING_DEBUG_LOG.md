# Head-Only Jigsaw Training Debug Log

## Results Summary
- Head-only training was unstable in `float16` but became stable in `bfloat16`.
- The first loss computation was finite; NaNs appeared after the first optimizer update, indicating update-time numeric instability in the fp16 path.
- An initial full `bfloat16` run completed all `35904 / 35904` steps, but validation performance was weak and plateaued by epoch 1.
- After restoring the important training settings under `bfloat16` (`learning_rate=1e-4`, `weight_decay=0.01`, capped `pos_weight`, fp32 logits in loss), head-only performance improved dramatically and remained stable through a full 2-epoch run.
- The later rerun plateaued between epoch 1 and epoch 2, suggesting the restored head-only setup is viable but saturates quickly.
- The post-training export bug was fixed by casting logits to fp32 before NumPy conversion.
- Export recovery succeeded on a `3090`: first on a 256-example smoke slice, then on the full validation split.
- The main operational blocker in export recovery was infrastructure selection / PVC usage, not remaining export logic bugs:
  - export-only OOMed on an `RTX 2080 Ti` (11GB VRAM),
  - and one full-export pod sat in `ContainerCreating` for hours while a wait pod was also mounted to the same PVC, suggesting PVC mount/attach contention or related startup delay.

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

### 17) Restored key training settings under `bfloat16`
- Reverted the temporary ultra-conservative training settings while keeping the `bfloat16` load path.
- Restored:
  - `learning_rate=1e-4`
  - `weight_decay=0.01`
  - capped `pos_weight` (still capped at `30.0`, not fully uncapped)
  - fp32 logits in the BCE loss path
  - `max_grad_norm=1.0`
- Kept excluded:
  - `fp16=True` in `TrainingArguments`
  - `model.score.float()`
- Purpose:
  - test whether `float16` had been the real cause of instability, rather than the restored optimization settings themselves

### 18) Restored-config head-only rerun stayed numerically stable
- Early debug output remained finite through at least `DEBUG step-call=19`
- Subsequent trainer logs showed:
  - finite loss values
  - finite `grad_norm`
  - no immediate `NaN` logits
  - no `Non-finite loss detected`
- Interpretation:
  - restoring capped `pos_weight` and the less conservative optimizer settings did not recreate the earlier `float16` instability
  - this strongly supports the conclusion that the original blow-up was primarily due to the `float16` path

### 19) Restored-config head-only rerun produced much stronger metrics
- Full rerun completed:
  - `35904 / 35904` steps
  - `train_runtime = 25199.8818` seconds (`6:59:59.88`)
- Epoch 1 eval:
  - `eval_loss = 0.6562253832817078`
  - `eval_micro_f1 = 0.5741170404743491`
  - `eval_macro_f1 = 0.4569263388199886`
  - `eval_samples_f1 = 0.04738087732823916`
  - `eval_subset_accuracy = 0.8776789071312194`
- Epoch 2 eval:
  - `eval_loss = 0.6344109177589417`
  - `eval_micro_f1 = 0.5734921439432337`
  - `eval_macro_f1 = 0.45521651647824873`
  - `eval_samples_f1 = 0.04825822998213022`
  - `eval_subset_accuracy = 0.8763002882566737`
- Relative to the earlier weak `bfloat16` head-only baseline:
  - `eval_micro_f1` improved from about `0.0735` to about `0.574`
  - `eval_macro_f1` improved from about `0.0694` to about `0.4569`
- Interpretation:
  - head-only is not inherently unusable on this task
  - the earlier weak result was heavily confounded by the stripped-down stability settings forced by the `float16` failures

### 20) Improved rerun still plateaued after epoch 1
- Epoch 2 slightly improved `eval_loss`, but headline F1 metrics were effectively flat relative to epoch 1.
- Interpretation:
  - the restored head-only setup learns a meaningful signal
  - but most of the useful learning appears to happen by the end of epoch 1
  - additional epochs alone are unlikely to be the main lever for further improvement

### 21) New post-training failure: export path with `bfloat16` logits
- Training and built-in evaluation completed successfully.
- The crash happened later during the optional prediction-export block:
  - in `infer_probabilities()`
  - at `probs = torch.sigmoid(logits).numpy()`
- Error:
  - `TypeError: Got unsupported ScalarType BFloat16`
- Consequence:
  - training artifacts and eval metrics were saved
  - but `/workspace/sequence_classification/predictions` was not created
  - so `val_predictions.csv`, `per_label_metrics.csv`, `overall_metrics.json`, and `thresholds.json` were not written
- Immediate fix:
  - cast logits to fp32 before NumPy conversion, e.g. `torch.sigmoid(logits.float()).numpy()`

### 22) Recovery work after the export failure
- Patched the head-only script to:
  - cast logits to fp32 before converting to NumPy
  - support `--export-only`
  - support `--saved-model-dir` so prediction artifacts can be regenerated from `/workspace/sequence_classification/saved_model` without retraining
- Verified that the following training artifacts already existed on PVC:
  - `saved_model/`
  - `checkpoint-17952/`
  - `checkpoint-35904/`
  - `eval_results.json`
  - `train_results.json`
  - `all_results.json`
  - `trainer_state.json`
- Confirmed that `predictions/` still did not exist after the failed run.
- Practical implication:
  - retraining is not required to recover prediction artifacts
  - only the post-training export step needs to be rerun successfully

### 23) Added export-debugging controls and logs
- Extended the head-only script to support:
  - `--export-limit`
  - `--export-batch-size`
- Added explicit export progress prints:
  - export device print
  - batch count print
  - periodic `Completed export batch ...` progress lines
- Skipped tokenized-dataset construction in `--export-only` mode.
- Also fixed an intermediate control-flow bug where export-only could still reference `tokenized` and crash with:
  - `UnboundLocalError: local variable 'tokenized' referenced before assignment`

### 24) Confirmed export-only OOM on `RTX 2080 Ti`
- Running export-only from a wait pod on an `RTX 2080 Ti` reproduced a clear hardware limit.
- Failure:
  - `torch.cuda.OutOfMemoryError`
- The failure occurred at:
  - `model = model.to(export_device)`
- Interpretation:
  - the patched export path was no longer silently failing
  - the remaining blocker on that pod was insufficient VRAM, not incorrect export logic

### 25) Smoke export succeeded on a `3090`
- A dedicated export job targeting `NVIDIA-GeForce-RTX-3090` was created.
- Smoke export config:
  - `--export-only`
  - `--saved-model-dir /workspace/sequence_classification/saved_model`
  - `--output-dir /workspace/sequence_classification`
  - `--export-limit 256`
  - `--export-batch-size 8`
- Smoke run completed successfully and wrote:
  - `predictions/overall_metrics.json`
  - `predictions/per_label_metrics.csv`
  - `predictions/thresholds.json`
  - `predictions/val_predictions.csv`
- Smoke metrics:
  - `micro_f1 = 0.5401459854014599`
  - `macro_f1 = 0.33289686370681076`
  - `samples_f1 = 0.052046130952380955`
  - `subset_accuracy = 0.875`
- Interpretation:
  - the smoke run validated the export path
  - but the slice was too small for meaningful rare-label conclusions (`threat` had zero positives; some labels had only one positive)


### 26) Full export recovery completed successfully
- Export log showed:
  - `Starting export inference over 15958 examples in 1995 batches (batch_size=8)`
  - completion through `1995/1995`
  - `Saved metrics: /workspace/sequence_classification/predictions`
- Final exported artifacts on PVC:
  - `overall_metrics.json`
  - `per_label_metrics.csv`
  - `thresholds.json`
  - `val_predictions.csv`
- Final exported overall metrics matched the successful epoch-2 eval:
  - `micro_f1 = 0.5734921439432337`
  - `macro_f1 = 0.45521651647824873`
  - `samples_f1 = 0.04825822998213022`
  - `subset_accuracy = 0.8763002882566737`
- Final per-label F1:
  - `toxic = 0.6485753052917232`
  - `obscene = 0.6088794926004228`
  - `insult = 0.5891387822270981`
  - `severe_toxic = 0.3356164383561644`
  - `identity_hate = 0.3325942350332594`
  - `threat = 0.21649484536082475`
- Practical implication:
  - the export recovery is complete
  - prediction artifacts are now available for explanation/probing analysis without retraining

---

## Key Findings

### Finding 1: head-only can be made stable
The instability was not ultimately caused by `pos_weight` alone. The decisive fix was moving from fp16 to bfloat16.

### Finding 2: the earlier weak head-only result was not the final story
Once the important training settings were restored under `bfloat16`, head-only performance improved dramatically:
- `eval_macro_f1` rose to about `0.457`
- `eval_micro_f1` rose to about `0.574`

This is the main modeling result from the restored-config rerun.

### Finding 3: restored head-only is viable but still plateaus quickly
The improved run learned a meaningful signal by epoch 1, but epoch 2 produced little additional F1 improvement. The setup appears useful, but quickly saturating.

### Finding 4: the remaining failure is now in post-training export, not training stability
This was true before recovery, but is no longer the current state.

The export issue has now been resolved:
- fp32 casting fixed the `bfloat16` to NumPy problem
- smoke export succeeded on a `3090`
- full export succeeded on a `3090`

The practical blocker turned out to be infrastructure:
- `2080 Ti` VRAM was insufficient for export-only model transfer
- a full-export pod spent hours in `ContainerCreating` while another pod was mounted to the same PVC, suggesting mount/attach contention rather than a remaining code bug
