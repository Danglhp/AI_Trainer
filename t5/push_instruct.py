import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi

def push_trained_model_to_hub():
    # 1. Configuration
    LOCAL_MODEL_PATH = "./phobert-vi-analysis"  # Root folder where model files are stored
    HF_REPO_NAME = "kienhoang123/Poem_Analysis_Instruct_VN"  # New repo name for instruct model
    DATASET_NAME = "kienhoang123/Vietnamese_Poem_Analysis_VN"
    MODEL_NAME = "google/flan-t5-small"  # Base model you used
    
    # 2. Load the model and tokenizer from local path
    print(f"Loading model from {LOCAL_MODEL_PATH}")
    model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
    
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
    - instruction-tuned
    - flan-t5
    datasets:
    - {DATASET_NAME}
    ---

    # Instruction-Tuned T5 Model for Vietnamese Poem Analysis

    This model was fine-tuned on {DATASET_NAME} to analyze Vietnamese poetry using an instruction-based approach.

    ## Model Details

    - **Base Model**: {MODEL_NAME}
    - **Training Data**: Vietnamese poem analysis dataset
    - **Tasks**: Extract emotion, metaphor, setting, motion, and prompt from Vietnamese poems

    ## Usage

    ```python
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("{HF_REPO_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained("{HF_REPO_NAME}")

    # Create an instruction-based input
    instruction = '''
    Below is an instruction that describes a task.
    ### Instruction:
    Generate emotion, metaphor, setting, motion and prompt in Vietnamese for the following content.
    ### Input:
    Your Vietnamese poem here
    ### Output:
    '''

    inputs = tokenizer(instruction, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
    ```

    The output is formatted as: "emotion ||| metaphor ||| setting ||| motion ||| prompt"
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
    
    print("Model pushed successfully to Hugging Face Hub!")

if __name__ == "__main__":
    push_trained_model_to_hub()