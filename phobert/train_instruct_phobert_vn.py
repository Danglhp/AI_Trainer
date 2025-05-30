#!/usr/bin/env python3
# train_phobert_instruct.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from typing import Dict, List, Optional, Union, Tuple

# Custom model that uses PhoBERT and adds a classification layer
class PhoBERTForPoetryAnalysis(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_model_name)
        
        # Enable gradient checkpointing for memory efficiency
        self.encoder.gradient_checkpointing_enable()
        
        # Hidden size from the encoder
        hidden_size = self.encoder.config.hidden_size
        
        # We'll use a simpler classification approach - each field gets a binary classifier
        self.emotion_classifier = nn.Linear(hidden_size, 1)
        self.metaphor_classifier = nn.Linear(hidden_size, 1)
        self.setting_classifier = nn.Linear(hidden_size, 1)
        self.motion_classifier = nn.Linear(hidden_size, 1)
        self.prompt_classifier = nn.Linear(hidden_size, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=None,
    ):
        # Handle input shape issues
        if input_ids.dim() > 2:
            input_ids = input_ids.squeeze(0)
        if attention_mask is not None and attention_mask.dim() > 2:
            attention_mask = attention_mask.squeeze(0)
            
        # Get encoder outputs
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the CLS token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each field
        emotion_logits = self.emotion_classifier(pooled_output)
        metaphor_logits = self.metaphor_classifier(pooled_output)
        setting_logits = self.setting_classifier(pooled_output)
        motion_logits = self.motion_classifier(pooled_output)
        prompt_logits = self.prompt_classifier(pooled_output)
        
        # Combine all logits
        all_logits = torch.cat([
            emotion_logits, metaphor_logits, setting_logits, 
            motion_logits, prompt_logits
        ], dim=1)
        
        loss = None
        if labels is not None:
            # Use binary cross entropy loss since we're predicting presence/absence
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(all_logits, labels)
        
        return {"loss": loss, "logits": all_logits} if loss is not None else {"logits": all_logits}

def main():
    # 1. Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 2. Configurations
    MODEL_NAME     = "vinai/phobert-base"   # Vietnamese PhoBERT
    DATASET_NAME   = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    OUTPUT_DIR     = "./phobert-vi-analysis"
    BATCH_SIZE     = 16  # Increased batch size
    NUM_EPOCHS     = 3
    MAX_INPUT_LEN  = 128  # Reduced max length for speed
    
    print(f"[INFO] Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PhoBERTForPoetryAnalysis(MODEL_NAME).to(device)

    # 4. Load dataset
    print(f"[INFO] Loading dataset: {DATASET_NAME}")
    raw = load_dataset(DATASET_NAME, split="train")
    splits = raw.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": splits["train"], "eval": splits["test"]})
    
    # Print sample data
    print("[INFO] Sample data from dataset:")
    sample = ds["train"][0]
    for key, value in sample.items():
        print(f"  {key}: {value}")

    # 5. Process inputs with instruction format
    INSTRUCTION = (
        "Nhiệm vụ: Tạo cảm xúc, ẩn dụ, bối cảnh, chuyển động và gợi ý cho nội dung sau.\n"
        "Nội dung: "
    )
    
    def preprocess(batch):
        # Create instruction prompt
        prompt = INSTRUCTION + batch["content"]
        
        # Tokenize the instruction prompt
        tokenized_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=MAX_INPUT_LEN,
            return_tensors=None
        )
        
        # Create binary labels for each field (1 if present, 0 if absent)
        labels = [
            1.0 if batch["emotion_vi"] else 0.0,
            1.0 if batch["metaphor_vi"] else 0.0,
            1.0 if batch["setting_vi"] else 0.0,
            1.0 if batch["motion_vi"] else 0.0,
            1.0 if batch["prompt_vi"] else 0.0
        ]
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("[INFO] Preprocessing dataset...")
    tok_ds = ds.map(
        preprocess,
        batched=False,
        remove_columns=ds["train"].column_names
    )
    
    # Print a processed example
    print("[INFO] Sample processed data:")
    sample_processed = tok_ds["train"][0]
    print(f"  input_ids (first 10): {sample_processed['input_ids'][:10]}")
    print(f"  attention_mask (first 10): {sample_processed['attention_mask'][:10]}")
    print(f"  labels: {sample_processed['labels']}")

    # 6. Data collator & TrainingArguments
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("[INFO] Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,  # Mixed precision for speed
        learning_rate=3e-5,
        warmup_ratio=0.1,  # Warm up for 10% of training
        logging_steps=100,
        eval_strategy="epoch",  # Evaluate once per epoch
        save_strategy="epoch",  # Save once per epoch
        gradient_accumulation_steps=4,  # Accumulate gradients for larger effective batch size
        dataloader_num_workers=4,  # Parallel data loading
        optim="adamw_torch",  # Efficient optimizer
        max_grad_norm=1.0,
        load_best_model_at_end=True,  # Load the best model at the end
        metric_for_best_model="loss",  # Use loss as the metric for the best model
        push_to_hub=False,
    )

    # 7. Trainer & Train
    print("[INFO] Setting up trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["eval"],
        data_collator=data_collator,
    )
    
    print("[INFO] Starting training...")
    trainer.train()
    
    print("[INFO] Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final-model"))
    print("[INFO] Training complete!")

if __name__ == "__main__":
    main()