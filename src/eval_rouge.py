import sys
import os
sys.path.append(os.path.abspath('code/function_vectors/src'))

import torch
import numpy as np
import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import load_dataset
from utils.intervention_utils import fv_intervention_natural_text
from rouge_score import rouge_scorer

def evaluate_summaries():
    model_name = 'gpt2-xl'
    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device='cpu')
    
    # Load test data
    print("Loading datasets...")
    cnn_dataset = load_dataset('cnn', root_data_dir='dataset_files', test_size=0.1, seed=42)
    xsum_dataset = load_dataset('xsum', root_data_dir='dataset_files', test_size=0.1, seed=42)
    
    # Load FVs
    cnn_fv = torch.load('dataset_files/cnn_fv.pt') # [n_layers, 1600]
    xsum_fv = torch.load('dataset_files/xsum_fv.pt') # [n_layers, 1600]
    
    # We use a middle-late layer for intervention, e.g., layer 24
    layer_idx = 24
    # Calculate a difference vector to represent the "extractive" constraint
    # Or just use the activation directly. We will try a scaled difference.
    diff_fv = (cnn_fv[layer_idx] - xsum_fv[layer_idx]).unsqueeze(0) # [1, 1600]
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def eval_on_dataset(dataset, name, n_samples=5):
        print(f"\nEvaluating on {name}...")
        results = {"baseline": [], "intervention_cnn": [], "intervention_xsum": []}
        
        for i in range(n_samples):
            example = dataset['test'][i]
            article = example['input']
            reference = example['output']
            
            prompt = f"Article:\n{article}\n\nSummary:\n"
            
            # Baseline
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            baseline_out = model.generate(**inputs, max_new_tokens=30, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            baseline_text = tokenizer.decode(baseline_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Intervention with CNN FV (adding the difference towards CNN)
            intervention_fn_cnn = lambda out, name: out[0] + 0.1 * diff_fv.to(model.device) if isinstance(out, tuple) else out + 0.1 * diff_fv.to(model.device)
            # The baukit TraceDict handles this. Wait, fv_intervention_natural_text handles it.
            clean_out, interv_out_cnn = fv_intervention_natural_text(
                prompt, edit_layer=layer_idx, function_vector=0.5*diff_fv, 
                model=model, model_config=model_config, tokenizer=tokenizer, max_new_tokens=30
            )
            text_cnn = tokenizer.decode(interv_out_cnn[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

            clean_out, interv_out_xsum = fv_intervention_natural_text(
                prompt, edit_layer=layer_idx, function_vector=-0.5*diff_fv, 
                model=model, model_config=model_config, tokenizer=tokenizer, max_new_tokens=30
            )
            text_xsum = tokenizer.decode(interv_out_xsum[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            print(f"\n[Sample {i}]")
            print(f"Ref  : {reference}")
            print(f"Base : {baseline_text}")
            print(f"+CNN : {text_cnn}")
            print(f"+XSum: {text_xsum}")
            
            results["baseline"].append(scorer.score(reference, baseline_text)['rouge1'].fmeasure)
            results["intervention_cnn"].append(scorer.score(reference, text_cnn)['rouge1'].fmeasure)
            results["intervention_xsum"].append(scorer.score(reference, text_xsum)['rouge1'].fmeasure)
            
        print(f"Mean ROUGE-1 ({name}):")
        print(f"Baseline: {np.mean(results['baseline']):.4f}")
        print(f"Interv +CNN: {np.mean(results['intervention_cnn']):.4f}")
        print(f"Interv +XSum: {np.mean(results['intervention_xsum']):.4f}")
        
    eval_on_dataset(cnn_dataset, "CNN/DailyMail", n_samples=5)

if __name__ == "__main__":
    evaluate_summaries()
