# Head-Only Jigsaw Results

## Run Summary
- Model: `meta-llama/Llama-3.1-8B`
- Task: multi-label Jigsaw toxicity classification (`6` labels)
- Training mode: head-only sequence classification
- Frozen: Llama backbone (`model.model`)
- Trainable: classifier head (`model.score`)
- Stable dtype: `torch.bfloat16`
- Final export status: successful

## Final Training Configuration
- `learning_rate = 1e-4`
- `weight_decay = 0.01`
- capped `pos_weight` enabled
- `pos_weight` cap = `30.0`
- logits cast to fp32 in the BCE loss path
- `max_grad_norm = 1.0`
- `load_best_model_at_end = False`

## Runtime
- `train_runtime = 25199.8818` seconds
- approximately `6h 59m 59.9s`
- `eval_runtime` at epoch 2 = `1233.5029` seconds (`20m 33.5s`)

## Best Validation Metrics

### Epoch 1
- `eval_loss = 0.6562253832817078`
- `eval_micro_f1 = 0.5741170404743491`
- `eval_macro_f1 = 0.4569263388199886`
- `eval_samples_f1 = 0.04738087732823916`
- `eval_subset_accuracy = 0.8776789071312194`

### Epoch 2
- `eval_loss = 0.6344109177589417`
- `eval_micro_f1 = 0.5734921439432337`
- `eval_macro_f1 = 0.45521651647824873`
- `eval_samples_f1 = 0.04825822998213022`
- `eval_subset_accuracy = 0.8763002882566737`

## Final Exported Overall Metrics

These are the metrics written to:
- `/workspace/sequence_classification/predictions/overall_metrics.json`

They match the successful epoch-2 validation metrics.

- `micro_f1 = 0.5734921439432337`
- `macro_f1 = 0.45521651647824873`
- `samples_f1 = 0.04825822998213022`
- `subset_accuracy = 0.8763002882566737`

## Final Per-Label Metrics

These are the metrics written to:
- `/workspace/sequence_classification/predictions/per_label_metrics.csv`

| Label | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| `toxic` | `0.6737138830162086` | `0.6252452583387835` | `0.6485753052917232` | `1529` |
| `obscene` | `0.5501432664756447` | `0.6816568047337278` | `0.6088794926004228` | `845` |
| `insult` | `0.518840579710145` | `0.6814720812182741` | `0.5891387822270981` | `788` |
| `severe_toxic` | `0.23113207547169812` | `0.6125` | `0.3356164383561644` | `160` |
| `identity_hate` | `0.24193548387096775` | `0.5319148936170213` | `0.3325942350332594` | `141` |
| `threat` | `0.14383561643835616` | `0.4375` | `0.21649484536082475` | `48` |

## ROC AUC

Recovered from:
- `/workspace/sequence_classification/predictions/val_predictions.csv`
- `compexp_toxicity/head_only/recover_roc_auc_from_predictions.py`

Competition-aligned metric:
- `mean column-wise ROC AUC (macro ROC AUC) = 0.9506788333333333`

Additional overall ROC AUC:
- `micro ROC AUC = 0.958747`

### Per-Label ROC AUC

| Label | ROC AUC |
| --- | ---: |
| `toxic` | `0.939487` |
| `severe_toxic` | `0.950576` |
| `obscene` | `0.952893` |
| `threat` | `0.953350` |
| `insult` | `0.949603` |
| `identity_hate` | `0.958164` |

## Threshold Analysis

Confirmed exported prediction threshold:
- `/workspace/sequence_classification/predictions/thresholds.json`
- `type = global`
- `threshold = 0.5`

Per-label threshold sweep recovered from:
- `/workspace/sequence_classification/predictions/val_predictions.csv`
- `compexp_toxicity/head_only/optimize_thresholds_from_predictions.py`

### Overall Metrics With Per-Label Tuned Thresholds
- `micro_f1 = 0.5919019712306872`
- `macro_f1 = 0.4696405405278708`
- `samples_f1 = 0.05238846669477074`
- `subset_accuracy = 0.8768642687053515`

### Best Per-Label Thresholds And Metrics

| Label | Threshold | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: | ---: |
| `identity_hate` | `0.66` | `0.294118` | `0.425532` | `0.347826` | `141` |
| `insult` | `0.73` | `0.612352` | `0.591371` | `0.601679` | `788` |
| `obscene` | `0.65` | `0.615295` | `0.628402` | `0.621780` | `845` |
| `severe_toxic` | `0.60` | `0.263006` | `0.568750` | `0.359684` | `160` |
| `threat` | `0.49` | `0.151316` | `0.479167` | `0.230000` | `48` |
| `toxic` | `0.37` | `0.628965` | `0.687377` | `0.656875` | `1529` |

### Accuracy Context

Validation label-set balance:
- total examples = `15958`
- all-negative examples = `14334` (`89.82%`)
- any-positive examples = `1624` (`10.18%`)

Per-label accuracy versus always-negative baseline:

| Label | Pos % | Neg % | Always-Negative Acc | Acc @ 0.5 | Acc @ Tuned |
| --- | ---: | ---: | ---: | ---: | ---: |
| `identity_hate` | `0.88%` | `99.12%` | `99.12%` | `98.11%` | `98.59%` |
| `insult` | `4.94%` | `95.06%` | `95.06%` | `95.31%` | `96.13%` |
| `obscene` | `5.30%` | `94.70%` | `94.70%` | `95.36%` | `95.95%` |
| `severe_toxic` | `1.00%` | `99.00%` | `99.00%` | `97.57%` | `97.97%` |
| `threat` | `0.30%` | `99.70%` | `99.70%` | `99.05%` | `99.04%` |
| `toxic` | `9.58%` | `90.42%` | `90.42%` | `93.51%` | `93.12%` |

Prevalence-based random baseline context:
- for a class-prior random predictor, expected per-label `precision`, `recall`, and `F1` are approximately the label positive rate
- all observed per-label F1 scores are substantially above those prevalence baselines

## Export Artifacts
- `overall_metrics.json`
- `per_label_metrics.csv`
- `thresholds.json`
- `val_predictions.csv`

All were successfully written under:
- `/workspace/sequence_classification/predictions`

## Interpretation
- Head-only is viable on this task under the restored `bfloat16` configuration.
- Most useful learning happened by epoch 1; epoch 2 changed loss slightly but left headline F1 metrics nearly flat.
- The model is strongest on `toxic`, `obscene`, and `insult`.
- Rare labels remain substantially weaker, especially `threat`.
- Strong per-label ROC AUCs indicate meaningful positive-vs-negative separability for all six labels, even where thresholded accuracy is distorted by class imbalance.
- Threshold tuning improves metrics modestly, not dramatically; the main limitation is not just the default `0.5` cutoff.
- `subset_accuracy` is not a reliable headline metric here because nearly `90%` of validation examples are fully negative.
- The recovered baseline is strong enough to support probing / explanation work without retraining, but it should not be framed as a strong standalone classifier across all rare labels.

## Operational Notes
- The export bug was fixed by casting logits to fp32 before NumPy conversion.
- Export-only OOMed on an `RTX 2080 Ti` due to insufficient VRAM.
- Smoke export and full export both succeeded on a `3090`.
- A long `ContainerCreating` delay during full export is most plausibly explained by PVC mount/attach contention while a wait pod was also mounted to the same PVC.
