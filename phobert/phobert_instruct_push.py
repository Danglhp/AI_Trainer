import os
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import HfApi
import torch
import glob

def push_trained_model_to_hub():
    # 1. Configuration
    LOCAL_MODEL_PATH = "./phobert-vi-analysis/final-model"  # Path to saved model
    HF_REPO_NAME = "kienhoang123/PhoBERT_Poem_Analysis_Instruct"  # Repository name
    DATASET_NAME = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    MODEL_NAME = "vinai/phobert-base"  # Base model you used
    
    # 2. Check what files exist in the model directory
    print(f"Checking files in {LOCAL_MODEL_PATH}")
    if os.path.exists(LOCAL_MODEL_PATH):
        files = os.listdir(LOCAL_MODEL_PATH)
        print(f"Files found: {files}")
        
        # Look for common model file patterns
        model_files = []
        for pattern in ["*.bin", "*.pt", "*.pth", "*.safetensors"]:
            model_files.extend(glob.glob(os.path.join(LOCAL_MODEL_PATH, pattern)))
        print(f"Model files found: {model_files}")
    else:
        print(f"Directory {LOCAL_MODEL_PATH} does not exist!")
        return
    
    # Import the custom model class
    from train_instruct_phobert_vn import PhoBERTForPoetryAnalysis
    
    # 3. Try different loading approaches
    try:
        # Method 1: Try loading tokenizer from local path
        print("Attempting to load tokenizer from local path...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        print("Tokenizer loaded successfully from local path")
    except:
        print("Failed to load tokenizer from local path, using base model tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Try to load the model
    model = PhoBERTForPoetryAnalysis(MODEL_NAME)
    
    # Try different file names for the model weights
    possible_model_files = [
        "model.safetensors",
        "pytorch_model.safetensors", 
        "pytorch_model.bin",
        "model.bin", 
        "model.pt",
        "model.pth",
        "best_model.bin",
        "best_model.pt",
        "final_model.bin",
        "final_model.pt"
    ]
    
    model_loaded = False
    for model_file in possible_model_files:
        model_path = os.path.join(LOCAL_MODEL_PATH, model_file)
        if os.path.exists(model_path):
            print(f"Found model file: {model_path}")
            try:
                # Load the trained weights - handle both .bin and .safetensors
                if model_file.endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_path)
                else:
                    state_dict = torch.load(model_path, map_location='cpu')
                
                # Handle different state dict formats
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                elif 'state_dict' in state_dict:
                    model.load_state_dict(state_dict['state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                print(f"Model weights loaded successfully from {model_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Failed to load {model_file}: {e}")
                continue
    
    if not model_loaded:
        print("Could not load model weights from any expected file!")
        print("Please check your model saving code and ensure the model is saved properly.")
        return
    
    # 4. Save the model in the correct format for Hugging Face
    print("Preparing model for Hugging Face Hub...")
    
    # Create a temporary directory to save the model in HF format
    temp_model_path = "./temp_hf_model"
    os.makedirs(temp_model_path, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(temp_model_path)
    
    # Save model - we need to save it as a proper Hugging Face model
    # Since this is a custom architecture, we'll save the state dict
    torch.save(model.state_dict(), os.path.join(temp_model_path, "pytorch_model.bin"))
    
    # Create config.json for the model
    config = {
        "architectures": ["PhoBERTForPoetryAnalysis"],
        "model_type": "phobert",
        "base_model": MODEL_NAME,
        "task_specific_params": {
            "classification_heads": ["emotion", "metaphor", "setting", "motion", "prompt"]
        }
    }
    
    import json
    with open(os.path.join(temp_model_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # 5. Push to Hub using the API (since we have a custom architecture)
    print(f"Pushing model to Hugging Face Hub: {HF_REPO_NAME}")
    
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=HF_REPO_NAME, exist_ok=True)
        print(f"Repository {HF_REPO_NAME} created/verified")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload all files in the temp directory
    api.upload_folder(
        folder_path=temp_model_path,
        repo_id=HF_REPO_NAME,
        repo_type="model"
    )
    
    # 6. Create and push a comprehensive model card
    model_card = f"""---
language: vi
license: apache-2.0
tags:
- vietnamese
- poem-analysis
- phobert
- classification
- poetry
datasets:
- {DATASET_NAME}
base_model: {MODEL_NAME}
---

# PhoBERT Model for Vietnamese Poem Analysis

This model was fine-tuned from {MODEL_NAME} on {DATASET_NAME} to analyze Vietnamese poetry across multiple dimensions.

## Model Details

- **Base Model**: {MODEL_NAME}
- **Training Data**: Vietnamese poem analysis dataset
- **Architecture**: Custom PhoBERT with multiple classification heads
- **Tasks**: Multi-label classification for:
  - Emotion detection
  - Metaphor identification  
  - Setting analysis
  - Motion detection
  - Prompt presence

## Model Architecture

The model extends PhoBERT with 5 binary classification heads, each predicting the presence/absence of specific poetic elements.

## Usage

⚠️ **Important**: This model uses a custom architecture. You need to define the model class before loading:

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class PhoBERTForPoetryAnalysis(nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Classification heads
        self.emotion_classifier = nn.Linear(hidden_size, 1)
        self.metaphor_classifier = nn.Linear(hidden_size, 1)
        self.setting_classifier = nn.Linear(hidden_size, 1)
        self.motion_classifier = nn.Linear(hidden_size, 1)
        self.prompt_classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        emotion_logits = self.emotion_classifier(pooled_output)
        metaphor_logits = self.metaphor_classifier(pooled_output)
        setting_logits = self.setting_classifier(pooled_output)
        motion_logits = self.motion_classifier(pooled_output)
        prompt_logits = self.prompt_classifier(pooled_output)
        
        all_logits = torch.cat([
            emotion_logits, metaphor_logits, setting_logits, 
            motion_logits, prompt_logits
        ], dim=1)
        
        return {{"logits": all_logits}}

# Load the model
tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_NAME}")
model = PhoBERTForPoetryAnalysis("{MODEL_NAME}")

# Load the fine-tuned weights
model.load_state_dict(torch.load("pytorch_model.bin", map_location='cpu'))
model.eval()

# Example usage
poem = "Your Vietnamese poem here"
instruction = "Nhiệm vụ: Tạo cảm xúc, ẩn dụ, bối cảnh, chuyển động và gợi ý cho nội dung sau.\\nNội dung: " + poem

inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    
logits = outputs["logits"]
predictions = torch.sigmoid(logits) > 0.5

# Interpret results
fields = ["emotion", "metaphor", "setting", "motion", "prompt"]
results = {{}}
for i, field in enumerate(fields):
    results[field] = predictions[0][i].item()
    
print(results)
```

## Training Details

- **Base Model**: {MODEL_NAME}
- **Fine-tuning approach**: Multi-task learning with binary classification heads
- **Input format**: Instruction + poem content
- **Output**: Binary predictions for 5 poetic elements

## Citation

If you use this model, please cite the original PhoBERT paper:

```bibtex
@inproceedings{{phobert,
    title = {{PhoBERT: Pre-trained language models for Vietnamese}},
    author = {{Dat Quoc Nguyen and Anh Tuan Nguyen}},
    booktitle = {{Findings of the Association for Computational Linguistics: EMNLP 2020}},
    year = {{2020}},
    pages = {{1037--1042}}
}}
```
"""
    
    # Write and upload model card
    with open(os.path.join(temp_model_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(model_card)
    
    api.upload_file(
        path_or_fileobj=os.path.join(temp_model_path, "README.md"),
        path_in_repo="README.md",
        repo_id=HF_REPO_NAME,
    )
    
    print("✅ PhoBERT model pushed successfully to Hugging Face Hub!")
    print(f"🔗 Model available at: https://huggingface.co/{HF_REPO_NAME}")
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_model_path)
    print("🧹 Temporary files cleaned up")

if __name__ == "__main__":
    push_trained_model_to_hub()