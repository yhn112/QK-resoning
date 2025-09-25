# Think First, Then Select and Verify with Query–Key Alignment

Anonymous repository with code for experiments from the research paper.

We demonstrate that Chain-of-Thought (CoT) prompting strengthens internal Query-Key (QK) alignment, enabling answer selection and verification directly from model activations rather than decoded tokens. The method achieves up to ~22% performance gains across benchmarks by turning CoT into a deliberation-then-selection mechanism.

## Experiments

**QK-score with CoT for MCQA:**
- [`HLE_MCQA_qwen3_14b.ipynb`](HLE_MCQA_qwen3_14b.ipynb) - MCQA vs MCQA+CoT experiments on MMLU-Pro and HLE datasets

**QK-score for verification:**
- [`qk_verif_8B.py`](qk_verif_8B.py) - Verification of LLM solution correctness on MATH-500 and HLE

**Hypothesis selection:**
- [`llm_eval.py`](llm_eval.py) - Framework for candidate generation and QK-based selection
- [`models_local.py`](models_local.py) - Local model wrapper with optimized inference  
- [`utils.py`](utils.py) - Utilities for answer extraction and prompt templates
- [`qwen3.ipynb`](qwen3.ipynb) - Candidate generation experiments with self-consistency baseline


**Requirements:** PyTorch, Transformers, NumPy, Pandas, Datasets, VLLM
