# Literature Review: Function Vectors for Summarization

## Research Area Overview
This research investigates the intersection of **representation engineering** and **abstractive summarization**. Specifically, it explores whether the "Function Vector" (FV) mechanism—previously identified in simple in-context learning (ICL) tasks—can be generalized to the more complex and constraint-driven task of summarization. The goal is to decode the internal "default constraints" (such as importance, length, and style) that LLMs apply when summarizing text.

## Key Papers

### 1. Function Vectors in Large Language Models (Todd et al., 2023)
- **Authors**: Eric Todd, Millicent L. Li, et al.
- **Key Contribution**: Identified that LLMs represent tasks as compact "function vectors" in their hidden states during ICL.
- **Methodology**: Used causal mediation analysis to identify attention heads that mediate task execution. Summing the outputs of these heads creates the FV.
- **Relevance**: Provides the foundational mechanism for task representation that this research aims to generalize to summarization.

### 2. What Matters to an LLM? Behavioral and Computational Evidences from Summarization (Zhou et al., 2026)
- **Authors**: Yongxin Zhou, Changshun Wu, et al.
- **Key Contribution**: Investigated the internal notion of "importance" in LLM summarization.
- **Methodology**: Combined behavioral analysis (generating length-controlled summaries) with computational probing and attention analysis.
- **Results**: Identified that middle-to-late layers are highly predictive of what the model considers "important."
- **Relevance**: Directly supports the hypothesis that summarization constraints (like salience) are internally represented and can be decoded.

### 3. Length Representations in Large Language Models (Moon et al., 2025)
- **Authors**: Sangjun Moon, Dasom Choi, et al.
- **Key Contribution**: Discovered internal representations of output sequence length in LLMs.
- **Methodology**: Scaling specific hidden units allowed control over the output length.
- **Results**: Showed that length information is partially disentangled from semantic information.
- **Relevance**: Provides a concrete example of a summarization-related constraint (length) that is represented as a manipulatable vector/unit.

### 4. Vector-ICL: In-context Learning with Continuous Vector Representations (Zhuang et al., 2024)
- **Authors**: Zhuang et al.
- **Key Contribution**: Explored ICL using continuous vectors instead of discrete tokens.
- **Relevance**: Demonstrates that complex tasks, including summarization, can be triggered or guided by vector-based representations of the task context.

## Common Methodologies
1. **Causal Mediation Analysis**: Identifying which internal components (attention heads, layers) are causally responsible for a specific behavior.
2. **Representation Probing**: Training linear probes on hidden states to predict task-related features (e.g., importance, length).
3. **Activation Intervention**: Modifying hidden states (e.g., adding a task vector) to induce or change model behavior.

## Standard Baselines
- **Standard ICL**: Few-shot summarization without explicit vector extraction.
- **Zero-shot Summarization**: Default model behavior.
- **Extractive Baselines**: Lead-3 (first 3 sentences), TextRank (for importance comparison).

## Evaluation Metrics
- **ROUGE (1, 2, L)**: Standard for summarization quality.
- **Faithfulness Metrics**: FactCC, QAGS (to check if FVs maintain or improve factuality).
- **Vector Cosine Similarity**: To compare FVs extracted from different contexts or domains.

## Gaps and Opportunities
- **Complexity of Summarization**: While Todd et al. focused on simple tasks (e.g., antonyms), summarization involves multiple simultaneous constraints (salience, brevity, coherence).
- **Decoding "Default" Constraints**: Most work focuses on *controlling* the model. There is an opportunity to use FVs to *understand* what a model naturally prioritizes when no explicit instructions are given.

## Recommendations for Our Experiment
1. **Dataset Selection**: Use CNN/Daily Mail (extractive bias) and XSum (abstractive bias) to see if FVs capture these differing "default" styles.
2. **Layer-wise Analysis**: Focus on middle-to-late layers for FV extraction, as suggested by Zhou et al. (2026).
3. **Constraint Decoding**: Extract FVs from prompts with different explicit constraints (e.g., "summarize in one sentence" vs "summarize in detail") and compare them to the "default" summarization FV.
