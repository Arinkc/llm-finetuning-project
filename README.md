# PyDocLlama — Fine-Tuning Llama 3.1 for High-Quality Python Docstrings

## What This Project Does
This project fine-tunes Meta's Llama 3.1 8B Instruct model to generate concise, 
PEP 257-compliant Python docstrings from raw function code, using QLoRA on a 
curated subset of the CodeSearchNet dataset.

## Why This Domain
I chose code documentation specifically because the input/output structure is well-defined and the evaluation is tractable. I can read outputs and tell if they're good, and I can write automated linters to check format compliance. That made it possible to focus on the fine-tuning technique itself without getting lost in subjective evaluation. I also wanted to understand whether targeted fine-tuning could fix the most common failure modes, and learn the full fine-tuning stack in the process.

## The Problem I'm Trying to Solve
Out of the box, Llama 3.1 8B Instruct can generate Python docstrings, but it 
exhibits several problems:
- Inconsistent formatting (mixes Google-style, NumPy-style, and freeform)
- Verbose output with unnecessary preamble like "Here is a docstring for your function:"
- Doesn't reliably follow PEP 257 conventions (one-line summary, blank line, details)
- Often hallucinates parameter descriptions for parameters that don't exist

The fine-tuned model should produce clean, consistent, PEP 257-compliant 
docstrings in a single style, without conversational filler, even for functions 
it has never seen before.

## Approach
- **Base Model:** meta-llama/Llama-3.1-8B-Instruct
- **Dataset:** CodeSearchNet (Python subset), filtered from ~450K examples 
  down to a curated ~10K high-quality function/docstring pairs
- **Fine-Tuning Method:** QLoRA (4-bit NF4 quantization + LoRA adapters on 
  attention and MLP projections)
- **Training Environment:** Kaggle Notebooks (T4 GPU) / Google Colab
- **Experiment Tracking:** Weights & Biases
- **Deployment:** [vLLM]

## Success Criteria
1. **Format compliance:** ≥85% of generated docstrings pass an automated PEP 257 
   linter (pydocstyle) on a held-out test set, vs. base model's ~[X]%
2. **No regression:** Fine-tuned model retains ≥95% of base model performance 
   on a general benchmark (e.g., HumanEval or a small MMLU subset)
3. **Qualitative win rate:** In a side-by-side LLM-as-judge comparison (using 
   GPT-4 or Claude as judge) on 100 held-out examples, fine-tuned model wins 
   or ties ≥70% of the time

## Project Status
🚧 In progress 

## Stack
Python 3.10, PyTorch, Hugging Face Transformers, PEFT, TRL, BitsAndBytes, 
Datasets, Accelerate, Weights & Biases, [vLLM or Ollama], Gradio

## Dataset

**Source:** [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) Python subset (412,178 raw examples)

**Curation Pipeline:**
1. Multi-criteria filter for Google-style docstrings (length bounds, style 
   detection, quality heuristics) → 126,357 examples passed (30.66%)
2. Random sample of 25,000 examples with `seed=42` for reproducibility
3. AST-based docstring stripping to prevent input-output leakage 
   → 24,970 examples after filtering structurally-invalid examples
4. Formatted as Llama 3.1 chat conversations with system prompt
5. Split 90/5/5 into train/validation/test

**Final dataset published:** [huggingface.co/datasets/Arinkc/pydoc-llama-codesearchnet-curated](https://huggingface.co/datasets/Arinkc/pydoc-llama-codesearchnet-curated)

| Split | Examples | Token Length p50 | Token Length p99 |
|-------|----------|------------------|------------------|
| Train | 22,473 | 201 | 514 |
| Validation | 1,248 | — | — |
| Test | 1,249 | — | — |

The short token lengths (p99 = 514) allow training with `max_seq_length=1024`, 
roughly halving GPU memory requirements compared to the standard 2048-token 
training context.

## Repository Structure
.
├── data/              # Raw and processed datasets (gitignored)
├── notebooks/         # Exploration and prototyping notebooks
├── src/               # Training and inference scripts
├── configs/           # Training hyperparameter configs
├── evaluation/        # Evaluation scripts and results
└── README.md

## Timeline
- **Week 1:** Setup, data exploration, data curation and formatting
- **Week 2:** Training pipeline, initial runs, hyperparameter sweeps
- **Week 3:** Evaluation suite, deployment, demo
- **Week 4:** Documentation, blog post, polish

## Results
*Coming soon — will include training curves, before/after comparison table, 
benchmark scores, and live demo link.*

## License
MIT — see LICENSE file.

## Acknowledgments
- Meta AI for releasing Llama 3.1
- The CodeSearchNet team for the dataset
- Hugging Face for the PEFT and TRL libraries
