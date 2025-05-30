import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import HfApi
import torch

def push_trained_model_to_hub():
    # 1. Configuration
    LOCAL_MODEL_PATH = "./phobert-vi-poem-analysis/final-model"  # Path to saved model
    HF_REPO_NAME = "kienhoang123/PhoBERT_Poem_Analysis_Seq2Seq"  # Repository name
    DATASET_NAME = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    MODEL_NAME = "vinai/phobert-base"  # Base model you used
    
    # 2. Load the model and tokenizer from local path
    print(f"Loading model from {LOCAL_MODEL_PATH}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH, 
        local_files_only=True,
        num_labels=5  # Make sure this matches what you used during training
    )
    
    # 3. Push to Hub
    print(f"Pushing model to Hugging Face Hub: {HF_REPO_NAME}")
    model.push_to_hub(HF_REPO_NAME)
    tokenizer.push_to_hub(HF_REPO_NAME)
    
    # 4. Create and push a model card
    api = HfApi()
    
    model_card = f"""---
        language: vi
        license: apache-2.0
        tags:
        - vietnamese
        - poem-analysis
        - phobert
        - sequence-classification
        datasets:
        - {DATASET_NAME}
        ---

        # PhoBERT Model for Vietnamese Poem Analysis

        This model was fine-tuned on {DATASET_NAME} to analyze Vietnamese poetry using a sequence classification approach.

        ## Model Details

        - **Base Model**: {MODEL_NAME}
        - **Training Data**: Vietnamese poem analysis dataset
        - **Tasks**: Predict presence of emotion, metaphor, setting, motion, and prompt in Vietnamese poems

        ## Usage

        ```python
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained("{HF_REPO_NAME}")

        # Prepare your input
        poem = "Your Vietnamese poem here"
        inputs = tokenizer(poem, return_tensors="pt", padding=True, truncation=True, max_length=256)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            
        logits = outputs.logits
        predictions = torch.sigmoid(logits) > 0.5  # Convert to binary predictions

        # Interpret results
        fields = ["emotion", "metaphor", "setting", "motion", "prompt"]
        for i, field in enumerate(fields):
            present = "present" if predictions[0][i].item() else "absent"
            print(f"{{field}}: {{present}}")
        """
        # Write model card to file
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(model_card)

    # Push the model card
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=HF_REPO_NAME,
    )

    print("PhoBERT model pushed successfully to Hugging Face Hub!")
    
if __name__ == "__main__":
    push_trained_model_to_hub()