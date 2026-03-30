# Resources Catalog: Function Vectors for Summarization

## Summary
This catalog lists all papers, datasets, and code repositories gathered for the research project "Function Vectors for Summarization."

## Papers
Total papers downloaded: 5

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Function Vectors in Large Language Models | Todd et al. | 2023 | papers/2310.15213_Function_Vectors.pdf | Foundational methodology. |
| What Matters to an LLM? | Zhou et al. | 2026 | papers/2602.00459_LLM_Summarization_Importance.pdf | Internal representations of salience. |
| Length Representations in LLMs | Moon et al. | 2025 | papers/2502.13524_Length_Representations.pdf | Decoding output length constraints. |
| Vector-ICL | Zhuang et al. | 2024 | papers/2402.05044_Vector_ICL.pdf | Summarization as a vector-based ICL task. |
| Linear Representations of Sentiment | Tigges et al. | 2023 | papers/2310.15154_Linear_Sentiment.pdf | Linear representation engineering in LLMs. |

## Datasets
Total datasets sampled: 3 + ICL Prompts

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| CNN/Daily Mail | HuggingFace | 100 samples | Summarization | datasets/cnn_dailymail/sample.csv | Lead-biased, extractive. |
| XSum | HuggingFace | 100 samples | Summarization | datasets/xsum/sample.csv | Highly abstractive. |
| BillSum | HuggingFace | 100 samples | Summarization | datasets/billsum/sample.csv | Legal domain. |
| ICL Prompts | Synthetic | 5 prompts | FV Extraction | datasets/icl_prompts.json | Formatted few-shot prompts. |

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| function_vectors | ericwtodd/function_vectors | Official Impl | code/function_vectors/ | Scripts for extraction and eval. |
| baukit | davidbau/baukit | Core Library | code/baukit/ | Dependency for causal mediation. |

## Recommendations for Experiment Design

1. **Primary Dataset**: Use **CNN/Daily Mail** and **XSum** as the primary contrastive pairs. Their inherent differences in "default" summarization style (extractive vs abstractive) will reveal if FVs can capture these high-level stylistic "constraints."
2. **Baseline Method**: Compare **Function Vector intervention** (adding the extracted FV to a zero-shot prompt) against **Few-Shot ICL** and **Zero-Shot summarization**.
3. **Evaluation Metric**: Use **ROUGE** for quality and **Cosine Similarity** between FVs extracted from different layers to identify the "optimal" task representation layer.
4. **Code to Reuse**: Adapt the `src/` directory from `code/function_vectors/` for extracting and intervening with FVs. Use `baukit` for hooking into the model activations.
