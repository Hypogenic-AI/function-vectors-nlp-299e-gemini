# Function Vectors for Summarization

## Overview
This project investigates the application of Representation Engineering—specifically Function Vectors (FVs)—to the complex task of text summarization. By extracting internal layer activations from `gpt2-xl` during few-shot summarization of CNN/Daily Mail (extractive) and XSum (abstractive) datasets, we decode how LLMs internally represent different summarization constraints.

## Key Findings
* **Unified Task Representation:** The internal function vectors for extractive and abstractive summarization are almost identical (Cosine Similarity > 0.93) across the first 95% of the model's layers.
* **Late-Stage Constraint Application:** The vectors diverge sharply at the final layer (Cosine Similarity drops to 0.38), proving that LLMs apply specific stylistic constraints (extractive vs abstractive) only at the very end of generation.
* **Intervention Insights:** Middle-layer interventions fail to steer summarization style, further validating that stylistic constraints are localized to late-layer representations.

## File Structure
* `REPORT.md`: Comprehensive research report with methodology, results, and analysis.
* `src/data_setup.py`: Script to preprocess CNN and XSum datasets.
* `src/extract.py`: Script to extract layer-wise function vectors and compute cosine similarities.
* `src/eval_rouge.py`: Script evaluating zero-shot intervention impact on generation using ROUGE.
* `dataset_files/`: Directory containing preprocessed datasets and extracted `.pt` function vectors.
* `code/function_vectors/`: The adapted original function vectors codebase.

## Reproduction
1. Initialize the virtual environment: `uv venv && source .venv/bin/activate`
2. Install dependencies: `uv add torch transformers datasets accelerate scikit-learn rouge-score evaluate pandas matplotlib seaborn bitsandbytes`
3. Install baukit: `uv pip install -e code/baukit/`
4. Setup data: `python src/data_setup.py`
5. Run extraction: `python src/extract.py`
6. Run evaluation: `python src/eval_rouge.py`