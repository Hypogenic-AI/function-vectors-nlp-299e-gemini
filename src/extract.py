import sys
import os
sys.path.append(os.path.abspath('code/function_vectors/src'))

import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.extract_utils import get_mean_layer_activations
from utils.model_utils import load_gpt_model_and_tokenizer
from utils.prompt_utils import load_dataset

def main():
    model_name = 'gpt2-xl' # 1.5B parameters, works well and fast
    print("Loading model...")
    model, tokenizer, model_config = load_gpt_model_and_tokenizer(model_name, device='cpu')
    
    prefixes = {"input": "Article:\n", "output": "Summary:\n", "instructions": "Summarize the following article.\n\n"}
    separators = {"input": "\n", "output": "\n\n", "instructions": ""}
    
    print("Loading CNN dataset...")
    cnn_dataset = load_dataset('cnn', root_data_dir='dataset_files', test_size=0.1, seed=42)
    print("Extracting CNN Function Vectors...")
    try:
        cnn_mean_activations = get_mean_layer_activations(
            cnn_dataset, model=model, model_config=model_config, tokenizer=tokenizer,
            n_icl_examples=2, N_TRIALS=10, prefixes=prefixes, separators=separators
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("Debugging shape...")
        from utils.extract_utils import gather_layer_activations
        from utils.prompt_utils import word_pairs_to_prompt_data
        word_pairs = cnn_dataset['train'][:2]
        word_pairs_test = cnn_dataset['valid'][:1]
        prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=word_pairs_test, prepend_bos_token=model_config['prepend_bos'], prefixes=prefixes, separators=separators)
        activations_td = gather_layer_activations(prompt_data=prompt_data, layers = model_config['layer_hook_names'], model=model, tokenizer=tokenizer, model_config=model_config)
        layer = model_config['layer_hook_names'][0]
        out = activations_td[layer].output
        print(f"Type of output: {type(out)}")
        if isinstance(out, tuple):
            print(f"Len of tuple: {len(out)}")
            print(f"Shape of out[0]: {out[0].shape}")
            if len(out[0].shape) == 2:
                print("Ah, output[0] is 2D!")
        else:
            print(f"Shape of out: {out.shape}")
        sys.exit(1)
    torch.save(cnn_mean_activations, 'dataset_files/cnn_fv.pt')
    
    print("Loading XSum dataset...")
    xsum_dataset = load_dataset('xsum', root_data_dir='dataset_files', test_size=0.1, seed=42)
    print("Extracting XSum Function Vectors...")
    xsum_mean_activations = get_mean_layer_activations(
        xsum_dataset, model=model, model_config=model_config, tokenizer=tokenizer,
        n_icl_examples=2, N_TRIALS=10, prefixes=prefixes, separators=separators
    )
    torch.save(xsum_mean_activations, 'dataset_files/xsum_fv.pt')
    
    print("Computing Cosine Similarity between CNN and XSum vectors across layers...")
    cos = torch.nn.CosineSimilarity(dim=0)
    for layer in range(model_config['n_layers']):
        sim = cos(cnn_mean_activations[layer], xsum_mean_activations[layer])
        print(f"Layer {layer}: Cosine Similarity = {sim.item():.4f}")

if __name__ == "__main__":
    main()
