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
- The recovered baseline is strong enough to support probing / explanation work without retraining.

## Operational Notes
- The export bug was fixed by casting logits to fp32 before NumPy conversion.
- Export-only OOMed on an `RTX 2080 Ti` due to insufficient VRAM.
- Smoke export and full export both succeeded on a `3090`.
- A long `ContainerCreating` delay during full export is most plausibly explained by PVC mount/attach contention while a wait pod was also mounted to the same PVC.
