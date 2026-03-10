import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "distilroberta-base"  # good starter model

label2id = {"entailment": 0, "contradiction": 1, "neutral": 2}
id2label = {v: k for k, v in label2id.items()}

# Toggle this:
# - True  -> small subset (CPU debug)
# - False -> full dataset (Nautilus run)
USE_SMALL_DEBUG = False


def main():
    print("Loading SNLI dataset...")
    toxicity = load_dataset("snli")

    # # SNLI labels: 0,1,2 are valid; -1 is "no label"
    # def filter_valid(example):
    #     return example["label"] != -1

    # snli = snli.filter(filter_valid)

    # if USE_SMALL_DEBUG:
    #     print("Using SMALL subset for quick debugging...")
    #     train_split = snli["train"].select(range(2000))
    #     val_split = snli["validation"].select(range(1000))
    # else:
    #     print("Using FULL train/validation splits...")
    #     train_split = snli["train"]
    #     val_split = snli["validation"]

    # print("Loading tokenizer and model...")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # def preprocess(batch):
    #     # We feed premise + hypothesis as sentence pair
    #     return tokenizer(
    #         batch["premise"],
    #         batch["hypothesis"],
    #         truncation=True,
    #         padding="max_length",
    #         max_length=128,
    #     )

    # print("Tokenizing...")
    # train_enc = train_split.map(preprocess, batched=True)
    # val_enc = val_split.map(preprocess, batched=True)

    # # Rename label -> labels for HF Trainer
    # train_enc = train_enc.rename_column("label", "labels")
    # val_enc = val_enc.rename_column("label", "labels")

    # # Only keep the columns the model actually needs
    # train_enc.set_format(
    #     type="torch",
    #     columns=["input_ids", "attention_mask", "labels"],
    # )
    # val_enc.set_format(
    #     type="torch",
    #     columns=["input_ids", "attention_mask", "labels"],
    # )

    # model = AutoModelForSequenceClassification.from_pretrained(
    #     MODEL_NAME,
    #     num_labels=3,
    #     id2label=id2label,
    #     label2id=label2id,
    # )

    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     preds = np.argmax(logits, axis=-1)
    #     acc = (preds == labels).astype(np.float32).mean().item()
    #     return {"accuracy": acc}

    # training_args = TrainingArguments(
    #     output_dir="./snli-finetuned-distilroberta",
    #     eval_strategy="epoch",            # evaluate every epoch
    #     save_strategy="epoch",            # save checkpoints every epoch
    #     logging_strategy="steps",
    #     logging_steps=100,
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=32,   # GPU can handle more than CPU
    #     per_device_eval_batch_size=64,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="accuracy",
    #     save_total_limit=2,
    #     report_to="none",
    #     fp16=True,                        # <--- this helps on GPU (Nautilus)
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_enc,
    #     eval_dataset=val_enc,
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics,
    # )

    # if USE_SMALL_DEBUG:
    #     print("Starting training (small subset)...")
    # else:
    #     print("Starting training (FULL dataset)...")

    # trainer.train()

    # print("Evaluating on validation set...")
    # metrics = trainer.evaluate()
    # print(metrics)

    # # Save final best model + tokenizer (for later eval / reuse)
    # save_dir = "./snli-finetuned-distilroberta/best"
    # print(f"Saving model to {save_dir}...")
    # trainer.save_model(save_dir)
    # tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
