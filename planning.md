## Research Question
Can function vectors be generalized to complex functions such as summarization, and if so, can we decode from them the default constraints (e.g., salience, length, style) that LLMs use in summarization?

## Motivation & Novelty Assessment

### Why This Research Matters
Understanding the internal representations that drive complex tasks like summarization is critical for building controllable and transparent LLMs. While previous work has shown that simple in-context learning tasks can be compressed into "function vectors," proving this for a multifaceted task like summarization would significantly advance representation engineering.

### Gap in Existing Work
Existing research on function vectors (Todd et al., 2023) focused primarily on simple tasks (e.g., antonyms, capitalization). While other works (Zhou et al., 2026; Moon et al., 2025) have explored the internal representations of specific constraints like salience and length independently, there is a gap in understanding whether a single, unified "function vector" can capture the holistic, complex constraints of default summarization styles (e.g., extractive vs. abstractive).

### Our Novel Contribution
We propose to extract function vectors for the summarization task and contrast them across datasets with different inherent styles (CNN/Daily Mail vs. XSum). We will not only test if these vectors can induce summarization in zero-shot contexts but also use them to decode and compare the "default constraints" (extractive vs. abstractive bias) represented internally by the model.

### Experiment Justification
- **Experiment 1 (Extraction & Intervention):** To verify if a summarization function vector can be reliably extracted and if injecting it induces summarization behavior comparable to standard ICL.
- **Experiment 2 (Decoding Default Constraints):** To compare function vectors derived from CNN/Daily Mail (extractive bias) vs. XSum (abstractive bias) to see if these stylistic constraints are linearly separable and decoded from the vectors.

## Proposed Methodology

### Approach
We will use causal mediation analysis and activation intervention (leveraging the `baukit` library) to extract task representations from the middle-to-late layers of an LLM during few-shot summarization. We will then inject these vectors into zero-shot prompts to evaluate their efficacy. Since function vectors require internal model access, we will use a small local open-source model (e.g., `gpt2` or a small Llama/Pythia model) suitable for CPU/GPU.

### Experimental Steps
1. **Environment & Data Setup:** Prepare CNN/Daily Mail and XSum datasets. Set up `baukit` and local environment.
2. **FV Extraction:** Extract function vectors from the model using few-shot prompts from both datasets.
3. **Intervention & Evaluation:** Inject the extracted FVs into zero-shot prompts. Generate summaries and compute ROUGE scores.
4. **Decoding & Analysis:** Compute cosine similarity between FVs from different datasets and perform PCA to see if we can separate the "extractive" vs. "abstractive" constraints.

### Baselines
- Zero-shot summarization (no FV intervention).
- Standard Few-shot ICL summarization.

### Evaluation Metrics
- ROUGE-1, ROUGE-2, ROUGE-L for summarization quality.
- Cosine similarity between extracted vectors to measure representation overlap.

### Statistical Analysis Plan
- T-tests to compare ROUGE scores between FV-intervention and zero-shot baselines.

## Expected Outcomes
- FV intervention should significantly outperform zero-shot summarization and approach few-shot performance.
- FVs from CNN/Daily Mail and XSum will show measurable differences (e.g., low cosine similarity or separability), reflecting the model's encoding of different default constraints (extractive vs. abstractive).

## Timeline and Milestones
- Phase 2 (Setup): 20 mins.
- Phase 3 (Implementation): 60 mins.
- Phase 4 (Experimentation): 60 mins.
- Phase 5 & 6 (Analysis and Documentation): 60 mins.

## Potential Challenges
- Small models might not be capable of high-quality summarization, making the FV effect hard to measure. Mitigation: Focus on simpler inputs or use a larger model if GPU is available.
- Extracting vectors for long contexts might be noisy. Mitigation: Use short documents for extraction.

## Success Criteria
- Successful extraction of a summarization FV that improves ROUGE scores over the zero-shot baseline.
- Meaningful differences found between FVs of different summarization styles.