import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import torch.nn as nn

class PoemAnalysisModel(nn.Module):
    def __init__(self, bert_model, num_labels=5):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Get CLS token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

def main():
    # 1. Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 2. Configurations
    MODEL_NAME     = "vinai/phobert-base"  # PhoBERT model for Vietnamese
    DATASET_NAME   = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    OUTPUT_DIR     = "./phobert-vi-poem-analysis"
    BATCH_SIZE     = 8
    NUM_EPOCHS     = 3
    MAX_INPUT_LEN  = 256  # PhoBERT's max context size

    # 3. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=5  # We'll output 5 values for each poem
    )
    model = base_model.to(device)

    # 4. Load your dataset from the Hub
    raw = load_dataset(DATASET_NAME, split="train")
    splits = raw.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": splits["train"], "eval": splits["test"]})

    # 5. Preprocessing
    def preprocess(batch):
        # Tokenize the content
        tokenized_inputs = tokenizer(
            batch["content"], 
            truncation=True, 
            padding="max_length", 
            max_length=MAX_INPUT_LEN
        )
        
        # Convert string labels to numeric values (normalize between 0-1)
        # Create a list of labels for each example
        labels = []
        for i in range(len(batch["content"])):
            example_labels = [
                float(1) if batch["emotion_vi"][i] else float(0),
                float(1) if batch["metaphor_vi"][i] else float(0),
                float(1) if batch["setting_vi"][i] else float(0), 
                float(1) if batch["motion_vi"][i] else float(0),
                float(1) if batch["prompt_vi"][i] else float(0)
            ]
            labels.append(example_labels)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names
    )

    # 6. Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        fp16=True,                     # mixed-precision on GPU
        logging_steps=100,
        eval_strategy="steps",   # This is the correct parameter name
        save_steps=500,
        push_to_hub=False,             # Set True to push checkpoint
    )

    # 8. Trainer
    trainer = Trainer(
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