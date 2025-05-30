---
        language: vi
        license: apache-2.0
        tags:
        - vietnamese
        - poem-analysis
        - phobert
        - sequence-classification
        datasets:
        - kienhoang123/Vietnamese_Poem_Analysis_VN
        ---

        # PhoBERT Model for Vietnamese Poem Analysis

        This model was fine-tuned on kienhoang123/Vietnamese_Poem_Analysis_VN to analyze Vietnamese poetry using a sequence classification approach.

        ## Model Details

        - **Base Model**: vinai/phobert-base
        - **Training Data**: Vietnamese poem analysis dataset
        - **Tasks**: Predict presence of emotion, metaphor, setting, motion, and prompt in Vietnamese poems

        ## Usage

        ```python
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("kienhoang123/PhoBERT_Poem_Analysis_Seq2Seq")
        model = AutoModelForSequenceClassification.from_pretrained("kienhoang123/PhoBERT_Poem_Analysis_Seq2Seq")

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
            print(f"{field}: {present}")
        