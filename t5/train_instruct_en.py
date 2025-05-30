#!/usr/bin/env python3
# train_instruction_hf.py

import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

def main():
    # 1. Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 2. Configurations
    MODEL_NAME     = "google/flan-t5-small"   # or another instruction-capable Seq2Seq
    DATASET_NAME   = "kienhoang123/Vietnamese_Poem_Analysis_EN"
    OUTPUT_DIR     = "./t5-en-instruct-hf"
    BATCH_SIZE     = 4
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

    # 5. Build instruction prompts
    INSTRUCTION = (
        "Below is an instruction that describes a task.\n"
        "### Instruction:\n"
        "Generate emotion, metaphor, setting, motion and prompt in Vietnamese for the following content.\n"
        "### Input:\n"
    )
    def make_if(batch):
        prompt = INSTRUCTION + batch["content"] + "\n### Output:\n"
        output = " ||| ".join([
            str(batch["emotion_en"] or ""),
            str(batch["metaphor_en"] or ""),
            str(batch["setting_en"] or ""),
            str(batch["motion_en"] or ""),
            str(batch["prompt_en"] or "")
        ])
        in_tok  = tokenizer(
            prompt, truncation=True, padding="max_length", max_length=MAX_INPUT_LEN
        )
        out_tok = tokenizer(
            output, truncation=True, padding="max_length", max_length=MAX_OUTPUT_LEN
        )
        in_tok["labels"] = out_tok.input_ids
        return in_tok

    tok_ds = ds.map(
        make_if,
        batched=False,
        remove_columns=ds["train"].column_names
    )

    # 6. Data collator & TrainingArguments
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        fp16=False,
        learning_rate=5e-5,  # Use a conservative learning rate
        warmup_steps=500,    # Gradual warmup
        logging_steps=100,
        eval_strategy="steps",
        save_steps=2000,
        max_grad_norm=1.0,
        push_to_hub=True,  # set True to auto-push
    )

    # 7. Trainer & Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final-model"))

if __name__ == "__main__":
    main()