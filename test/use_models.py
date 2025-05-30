import torch
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig

def load_seq2seq_model():
    # Load the PhoBERT model
    model_path = "kienhoang123/PhoBERT_Poem_Analysis_Seq2Seq"
    model = AutoModel.from_pretrained(model_path)  # Changed to AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_instruct_model():
    # Load the instruction-tuned model
    model_path = "kienhoang123/PhoBERT_Poem_Analysis_Instruct"
    try:
        # First try loading with AutoModel
        model = AutoModel.from_pretrained(model_path)
    except ValueError:
        print("Attempting to load with custom approach...")
        # Try an alternative - download the model files directly and create a custom wrapper
        try:
            # Import vinai/phobert-base as a fallback since PhoBERT is based on it
            from transformers import RobertaModel
            print("Loading as RoBERTa model since PhoBERT is based on RoBERTa architecture")
            model = RobertaModel.from_pretrained("vinai/phobert-base")
            print(f"Warning: Using base PhoBERT model instead of the fine-tuned version at {model_path}")
            print("\nTo use the actual fine-tuned model, please update your transformers library by running:")
            print("python e:\\AI Trainer\\test\\update_transformers.py")
        except Exception as e:
            print(f"Failed to load model: {e}")
            print("Please run: pip install --upgrade transformers")
            print("Or: pip install git+https://github.com/huggingface/transformers.git")
            raise
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except:
        # Fallback to vinai/phobert-base tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    return model, tokenizer

def generate_with_seq2seq(model, tokenizer, poem_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # For encoder-only models like RoBERTa, we need to encode the text and get embeddings
    inputs = tokenizer(poem_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state (embeddings)
    embeddings = outputs.last_hidden_state
    
    # For demonstration, we'll return a placeholder as we can't generate text with encoder-only models
    # In a real application, you'd need a separate decoder or classifier head
    print(f"Generated embeddings shape: {embeddings.shape}")
    return {"message": "This model is an encoder-only model and cannot directly generate text. Please implement a custom decoder or classifier."}

def generate_with_instruct(model, tokenizer, poem_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    instruction = (
        "Below is an instruction that describes a task.\n"
        "### Instruction:\n"
        "Generate emotion, metaphor, setting, motion and prompt in Vietnamese for the following content.\n"
        "### Input:\n"
        f"{poem_text}\n"
        "### Output:\n"
    )
    
    # For encoder-only models like RoBERTa, we need to encode the text and get embeddings
    inputs = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the last hidden state (embeddings)
    embeddings = outputs.last_hidden_state
    
    # For demonstration, we'll return a placeholder as we can't generate text with encoder-only models
    # In a real application, you'd need a separate decoder or classifier head
    print(f"Generated embeddings shape: {embeddings.shape}")
    return {"message": "This model is an encoder-only model and cannot directly generate text. Please implement a custom decoder or classifier."}

# Example usage
if __name__ == "__main__":
    # Test with a sample poem
    sample_poem = "Trên đầu súng giặc trăng như máu\nRừng cây im như những cột cao\nCao vút kinh hoàng\nRừng che bộ đội\nRừng vây quân thù."
    
    print("Testing Seq2Seq model:")
    seq2seq_model, seq2seq_tokenizer = load_seq2seq_model()
    seq2seq_result = generate_with_seq2seq(seq2seq_model, seq2seq_tokenizer, sample_poem)
    print(seq2seq_result)
    
    print("\nTesting Instruction model:")
    instruct_model, instruct_tokenizer = load_instruct_model()
    instruct_result = generate_with_instruct(instruct_model, instruct_tokenizer, sample_poem)
    print(instruct_result)