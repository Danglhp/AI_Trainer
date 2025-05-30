import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
import pandas as pd
import nltk
import os

# Ensure nltk packages are downloaded
nltk.download('punkt')

# Custom model class (must match the one used for training)
class PhoBERTForPoetryAnalysis(torch.nn.Module):
    def __init__(self, bert_model_name):
        super().__init__()
        from transformers import AutoModel
        import torch.nn as nn
        
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
        
        return {"logits": all_logits}

def format_prediction(prediction_dict):
    """Format the prediction dictionary into a string for ROUGE/BLEU evaluation"""
    result = []
    for key, value in prediction_dict.items():
        if value:
            result.append(f"{key}: Yes")
        else:
            result.append(f"{key}: No")
    return " ".join(result)

def format_ground_truth(example):
    """Format the ground truth into a string for ROUGE/BLEU evaluation"""
    result = []
    fields = ["emotion_vi", "metaphor_vi", "setting_vi", "motion_vi", "prompt_vi"]
    field_names = ["emotion", "metaphor", "setting", "motion", "prompt"]
    
    for i, field in enumerate(fields):
        if example[field]:
            result.append(f"{field_names[i]}: Yes")
        else:
            result.append(f"{field_names[i]}: No")
    return " ".join(result)

def main():
    # Configuration
    HF_MODEL_REPO = "kienhoang123/PhoBERT_Poem_Analysis_Instruct"
    BASE_MODEL = "vinai/phobert-base"
    DATASET_NAME = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer
    print(f"Loading tokenizer from {HF_MODEL_REPO}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO)
    except Exception as e:
        print(f"Failed to load tokenizer from HF repo: {e}")
        print("Loading tokenizer from base model instead")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load model
    print(f"Loading model from {HF_MODEL_REPO}")
    model = PhoBERTForPoetryAnalysis(BASE_MODEL)
    
    # Load weights from Hugging Face
    try:
        state_dict = torch.hub.load_state_dict_from_url(
            f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main/pytorch_model.bin",
            map_location=device
        )
        model.load_state_dict(state_dict)
        print("Model loaded successfully from Hugging Face")
    except Exception as e:
        print(f"Error loading model from Hugging Face: {e}")
        print("Please ensure the model is publicly available and the URL is correct")
        return
    
    model.to(device)
    model.eval()
    
    # Load test dataset
    print(f"Loading test dataset from {DATASET_NAME}")
    test_dataset = load_dataset(DATASET_NAME, split="train")
    # Use a smaller portion for testing
    test_dataset = test_dataset.select(range(min(100, len(test_dataset))))
    
    # Initialize ROUGE and BLEU metrics
    rouge = Rouge()
    
    # Store results
    results = []
    rouge_scores = []
    bleu_scores = []
    
    print(f"Running predictions on {len(test_dataset)} test examples...")
    
    # Process each test example
    for example in test_dataset:
        content = example["content"]
        instruction = f"Nhiệm vụ: Tạo cảm xúc, ẩn dụ, bối cảnh, chuyển động và gợi ý cho nội dung sau.\nNội dung: {content}"
          # Tokenize
        inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Remove token_type_ids if present (PhoBERT doesn't need it)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
            
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
        predictions = torch.sigmoid(logits) > 0.5
        
        # Convert predictions to dict
        fields = ["emotion", "metaphor", "setting", "motion", "prompt"]
        prediction_dict = {}
        for i, field in enumerate(fields):
            prediction_dict[field] = bool(predictions[0][i].item())
        
        # Format for comparison
        prediction_text = format_prediction(prediction_dict)
        ground_truth_text = format_ground_truth(example)
        
        # Calculate ROUGE scores
        try:
            rouge_score = rouge.get_scores(prediction_text, ground_truth_text)[0]
            rouge_scores.append(rouge_score)
        except Exception as e:
            print(f"Error calculating ROUGE: {e}")
            rouge_score = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}
        
        # Calculate BLEU score
        try:
            # Tokenize for BLEU
            prediction_tokens = nltk.word_tokenize(prediction_text.lower())
            reference_tokens = [nltk.word_tokenize(ground_truth_text.lower())]
            
            # Calculate BLEU score with smoothing
            smoothie = SmoothingFunction().method1
            bleu_score = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"Error calculating BLEU: {e}")
            bleu_score = 0
        
        # Store result
        result = {
            "content": content,
            "ground_truth": ground_truth_text,
            "prediction": prediction_text,
            "rouge-1": rouge_score["rouge-1"]["f"],
            "rouge-2": rouge_score["rouge-2"]["f"],
            "rouge-l": rouge_score["rouge-l"]["f"],
            "bleu": bleu_score
        }
        results.append(result)
    
    # Calculate average scores
    avg_rouge1 = np.mean([score["rouge-1"]["f"] for score in rouge_scores])
    avg_rouge2 = np.mean([score["rouge-2"]["f"] for score in rouge_scores])
    avg_rougel = np.mean([score["rouge-l"]["f"] for score in rouge_scores])
    avg_bleu = np.mean(bleu_scores)
    
    print(f"\nEvaluation Results:")
    print(f"Average ROUGE-1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-2: {avg_rouge2:.4f}")
    print(f"Average ROUGE-L: {avg_rougel:.4f}")
    print(f"Average BLEU: {avg_bleu:.4f}")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("evaluation_results", exist_ok=True)
    results_df.to_csv("evaluation_results/phobert_evaluation_results.csv", index=False)
    print("Detailed results saved to evaluation_results/phobert_evaluation_results.csv")

if __name__ == "__main__":
    main()