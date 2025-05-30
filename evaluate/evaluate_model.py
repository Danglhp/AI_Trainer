import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import numpy as np
from tqdm import tqdm
from use_models import generate_with_seq2seq, generate_with_instruct, load_seq2seq_model, load_instruct_model

# Download nltk data
nltk.download('punkt')

def prepare_references(sample):
    """Convert dataset sample to reference outputs."""
    # Combine all components with ||| separator
    reference = " ||| ".join([
        str(sample["emotion_vi"] or ""),
        str(sample["metaphor_vi"] or ""),
        str(sample["setting_vi"] or ""),
        str(sample["motion_vi"] or ""),
        str(sample["prompt_vi"] or "")
    ])
    # Tokenize for BLEU
    tokenized_ref = nltk.word_tokenize(reference.lower())
    return reference, tokenized_ref

def evaluate_model(model_type, test_samples=100):
    # Load dataset
    dataset = load_dataset("kienhoang123/Vietnamese_Poem_Analysis_VN", split="train")
    eval_samples = dataset.select(range(test_samples))
    
    # Initialize models
    if model_type == "seq2seq":
        model, tokenizer = load_seq2seq_model()
        generate_fn = generate_with_seq2seq
    else:  # instruction
        model, tokenizer = load_instruct_model()
        generate_fn = generate_with_instruct
    
    # Initialize metrics
    bleu_scores = []
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    # Initialize smoothing function for BLEU
    smooth = SmoothingFunction().method1
    
    # Evaluate each sample
    for sample in tqdm(eval_samples):
        # Get poem content
        poem = sample["content"]
        
        # Generate prediction
        result = generate_fn(model, tokenizer, poem)
        prediction = " ||| ".join([
            result.get("emotion", ""),
            result.get("metaphor", ""),
            result.get("setting", ""),
            result.get("motion", ""),
            result.get("prompt", "")
        ])
        
        # Prepare reference
        reference, tokenized_ref = prepare_references(sample)
        
        # Calculate BLEU
        tokenized_pred = nltk.word_tokenize(prediction.lower())
        if tokenized_pred:  # Only calculate if we have a prediction
            bleu = sentence_bleu([tokenized_ref], tokenized_pred, smoothing_function=smooth)
            bleu_scores.append(bleu)
        
        # Calculate ROUGE
        rouge_result = rouge_scorer_obj.score(reference, prediction)
        for metric in rouge_scores:
            rouge_scores[metric].append(rouge_result[metric].fmeasure)
    
    # Calculate average scores
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_rouge = {metric: np.mean(scores) if scores else 0 for metric, scores in rouge_scores.items()}
    
    return {
        'bleu': avg_bleu,
        'rouge1': avg_rouge['rouge1'],
        'rouge2': avg_rouge['rouge2'],
        'rougeL': avg_rouge['rougeL']
    }

if __name__ == "__main__":
    # Evaluate both models
    print("Evaluating Seq2Seq model...")
    seq2seq_metrics = evaluate_model("seq2seq")
    print(f"Seq2Seq Results:\n{seq2seq_metrics}")
    
    print("\nEvaluating Instruction model...")
    instruct_metrics = evaluate_model("instruct")
    print(f"Instruction Results:\n{instruct_metrics}")
    
    # Compare models
    print("\nModel Comparison:")
    metrics = ["bleu", "rouge1", "rouge2", "rougeL"]
    for metric in metrics:
        better = "Seq2Seq" if seq2seq_metrics[metric] > instruct_metrics[metric] else "Instruction"
        print(f"{metric}: Seq2Seq={seq2seq_metrics[metric]:.4f}, Instruction={instruct_metrics[metric]:.4f}, Better={better}")