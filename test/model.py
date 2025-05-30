import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_seq2seq_model():
    # Load the seq2seq model
    model_path = "kienhoang123/Poem_Analysis_Seq2Seq_VN"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def load_instruct_model():
    # Load the instruction-tuned model
    model_path = "kienhoang123/Poem_Analysis_Instruct_VN"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_with_seq2seq(model, tokenizer, poem_text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    inputs = tokenizer(poem_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the output
    components = result.split(" ||| ")
    if len(components) >= 5:
        return {
            "emotion": components[0],
            "metaphor": components[1],
            "setting": components[2],
            "motion": components[3],
            "prompt": components[4]
        }
    return {"raw_output": result}

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
    
    inputs = tokenizer(instruction, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Parse the output
    components = result.split(" ||| ")
    if len(components) >= 5:
        return {
            "emotion": components[0],
            "metaphor": components[1],
            "setting": components[2],
            "motion": components[3],
            "prompt": components[4]
        }
    return {"raw_output": result}

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