import os
import torch
import multiprocessing
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

def main():
    # -----------------------------------------------
    # 1. Thiết lập cấu hình chung
    # -----------------------------------------------
    MODEL_NAME      = "meta-llama/Llama-3.2-3B-Instruct"
    DATASET_NAME    = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    OUTPUT_DIR      = "outputs/llama-3.2-3B-vn-instruct"
    BATCH_SIZE      = 2            # Tăng batch nếu GPU đủ VRAM
    ACCUM_STEPS     = 2            # Giảm gradient_accumulation_steps để ít sub-steps hơn
    EPOCHS          = 3
    MAX_INPUT_LEN   = 384          # Giảm từ 512 xuống 384
    MAX_OUTPUT_LEN  = 64           # Giảm từ 128 xuống 64
    LEARNING_RATE   = 2e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {DEVICE}")

    # -----------------------------------------------
    # 2. Load tokenizer và model (8-bit Quant + LoRA)
    # -----------------------------------------------
    # 2.1. Cấu hình 8-bit
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # 2.2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # 2.3. Load model ở chế độ 8-bit
    model = LlamaForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
    )

    # 2.4. Resize embedding nếu thêm pad_token
    model.resize_token_embeddings(len(tokenizer))
    
    # Add this line to prepare the model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 2.5. Cấu hình LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    
    # Change this to False if you're using gradient checkpointing
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # In thông tin số lượng tham số trainable
    print("▶ LoRA modules:")
    model.print_trainable_parameters()

    # -----------------------------------------------
    # 3. Load và split dataset
    # -----------------------------------------------
    raw = load_dataset(DATASET_NAME, split="train")
    splits = raw.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": splits["train"], "eval": splits["test"]})

    print("▶ Columns:", ds["train"].column_names)

    # -----------------------------------------------
    # 4. Hàm định dạng instruction + input + output
    # -----------------------------------------------
    alpaca_template = """
    ### Instruction:
    Analyze the given poem and extract its emotional tone, metaphor, setting, motion, and generate a prompt based on the poem.

    ### Input Poem:
    {}

    ### Poem Analysis:
    Emotion: {}
    Metaphor: {}
    Setting: {}
    Motion: {}
    Prompt: {}
    """

    def preprocess_fn(batch):
        contents     = batch["content"]
        emotion_v    = batch["emotion_vi"]
        metaphor_v   = batch["metaphor_vi"]
        setting_v    = batch["setting_vi"]
        motion_v     = batch["motion_vi"]
        prompt_v     = batch["prompt_vi"]

        input_texts = []
        label_texts = []

        for c, e, mmp, s, mot, pr in zip(contents, emotion_v, metaphor_v, setting_v, motion_v, prompt_v):
            instruction_input = alpaca_template.format(c, e, mmp, s, mot, pr).strip()
            parts = instruction_input.split("### Poem Analysis:")
            inp_sentence = parts[0] + "### Poem Analysis:"
            lbl_sentence = parts[1].strip()
            input_texts.append(inp_sentence)
            label_texts.append(lbl_sentence)

        model_inputs = tokenizer(
            input_texts,
            truncation=True,
            max_length=MAX_INPUT_LEN,
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                label_texts,
                truncation=True,
                max_length=MAX_OUTPUT_LEN,
            )

        full_input_ids = []
        full_attention_mask = []
        labels_with_ignore = []

        for i in range(len(model_inputs["input_ids"])):
            input_ids_i = model_inputs["input_ids"][i]
            label_ids_i = labels["input_ids"][i]

            input_length = len(input_ids_i)
            label_length = len(label_ids_i)

            # Kết hợp input_ids + label_ids
            full_ids = input_ids_i + label_ids_i
            if len(full_ids) < MAX_INPUT_LEN + MAX_OUTPUT_LEN:
                full_ids += [tokenizer.pad_token_id] * (MAX_INPUT_LEN + MAX_OUTPUT_LEN - len(full_ids))
            full_ids = full_ids[: MAX_INPUT_LEN + MAX_OUTPUT_LEN]

            # Attention mask
            full_mask = [1] * (input_length + label_length)
            if len(full_mask) < MAX_INPUT_LEN + MAX_OUTPUT_LEN:
                full_mask += [0] * (MAX_INPUT_LEN + MAX_OUTPUT_LEN - len(full_mask))
            full_mask = full_mask[: MAX_INPUT_LEN + MAX_OUTPUT_LEN]

            # Labels with ignore index (-100) cho phần input
            label_mask = [-100] * input_length + label_ids_i
            if len(label_mask) < MAX_INPUT_LEN + MAX_OUTPUT_LEN:
                label_mask += [-100] * (MAX_INPUT_LEN + MAX_OUTPUT_LEN - len(label_mask))
            label_mask = label_mask[: MAX_INPUT_LEN + MAX_OUTPUT_LEN]

            full_input_ids.append(full_ids)
            full_attention_mask.append(full_mask)
            labels_with_ignore.append(label_mask)

        return {
            "input_ids": full_input_ids,
            "attention_mask": full_attention_mask,
            "labels": labels_with_ignore
        }

    tokenized_ds = ds.map(
        preprocess_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # -----------------------------------------------
    # 5. Data collator
    # -----------------------------------------------
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # -----------------------------------------------
    # 6. Cấu hình TrainingArguments
    # -----------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=300,
        save_steps=300,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_torch_fused",
        dataloader_num_workers=0,
        report_to="tensorboard",
        push_to_hub=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # -----------------------------------------------
    # 7. Khởi tạo Trainer và train
    # -----------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("▶ Training starts …")
    trainer.train()

    # -----------------------------------------------
    # 8. Lưu model và tokenizer đã fine-tune
    # -----------------------------------------------
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    print(f"[INFO] Fine-tuned model saved to {best_model_dir}")

if __name__ == "__main__":
    # This is needed on Windows when using multiprocessing
    multiprocessing.freeze_support()
    # Call the main function
    main()