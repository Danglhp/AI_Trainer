import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

def main():
    # 1. Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 2. Configurations
    MODEL_NAME     = "t5-small"                                   # or your preferred Seq2Seq checkpoint
    DATASET_NAME   = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    OUTPUT_DIR     = "./t5-vi-seq2seq-hf"
    BATCH_SIZE     = 8
    NUM_EPOCHS     = 3
    MAX_INPUT_LEN  = 512
    MAX_OUTPUT_LEN = 128

    # 3. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)

    # 4. Load your dataset from the Hub
    raw = load_dataset(DATASET_NAME, split="train")
    splits = raw.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": splits["train"], "eval": splits["test"]})

    # 5. Preprocessing
    def preprocess(batch):
        inputs = batch["content"]
        outputs = [
            " ||| ".join([
                str(ev or ""), 
                str(mg or ""), 
                str(sv or ""), 
                str(mv or ""), 
                str(pv or "")
            ]) for ev, mg, sv, mv, pv in zip(
                batch["emotion_vi"],
                batch["metaphor_vi"],
                batch["setting_vi"],
                batch["motion_vi"],
                batch["prompt_vi"]
            )
        ]
        model_inputs = tokenizer(
            inputs, truncation=True, padding="max_length", max_length=MAX_INPUT_LEN
        )
        labels = tokenizer(
            outputs, truncation=True, padding="max_length", max_length=MAX_OUTPUT_LEN
        ).input_ids

        model_inputs["labels"] = labels
        return model_inputs

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names
    )

    # 6. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 7. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,                        # mixed-precision on GPU
        logging_steps=100,
        eval_strategy="steps",            # Changed from evaluation_strategy
        save_steps=500,
        predict_with_generate=True,
        push_to_hub=False,                # set True to push checkpoint
    )

    # 8. Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 9. Train & Save
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final-model"))

if __name__ == "__main__":
    main()