# `RC` Inference

This folder contains code used for running `RC` decoding. Running this only requires vLLM (and its dependencies): if you do not intend to train models, you do not need to install verl.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Inference Strategies

The `inference/` directory contains the scripts for running `RC` and related baseines. All scripts use [vLLM](https://github.com/vllm-project/vllm) for high-throughput inference.

### 1. Standard Autoregressive Generation (`generate.py`)
Generates simple autoregressive rollouts for a given dataset and prompt.

**Sample Command:**
```bash
python -m inference.inference.generate \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/hmmt_2025_nov.json \
    --output_path outputs/hmmt_qwen3_4b.json \
    --prompt_path prompts/solution_generation_prompt.txt \
    --n_rollouts_per_problem 16 \
    --tp_size 4 \
    --temperature 0.7 \
    --top_p 0.8
```

### 2. `RC` decoding (`generate_complete.py`)
Implements `RC` decoding.
**Sample Command:**
```bash
python -m inference.inference.generate_complete \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/hmmt_2025_nov.json \
    --reasoning_prompt_path prompts/reasoning_prompt.txt \
    --summarization_prompt_path prompts/summarization_prompt.txt \
    --output_path outputs/hmmt_qwen3_4b_rc.json \
    --n 16 \
    --max_steps 12 \
    --tp_size 4
```

### 3. Multi-Model E2E (`generate_complete_multi_model.py`)
Similar to `generate_complete.py`, but allows using different models for the reasoning and summarization steps.

### 4. Reasoning without Summarization (`generate_no_summary.py`)
Baseline method: a multi-step reasoning loop where the model sees its previous attempt but no explicit summary is generated.

**Sample Command:**
```bash
python -m inference.inference.generate_no_summary \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/hmmt_2025_nov.json \
    --reasoning_prompt_path prompts/self_refine_sci_prompt.txt \
    --output_path outputs/hmmt_qwen3_4b_self_refine.json \
    --n 16 \
    --max_steps 12 \
    --tp_size 4
```

### 5. S1-style/Budget Forcing "Wait" Continuation (`generate_s1.py`)
Implements a budget forcing approach where the model is prompted to "Wait, let me continue thinking" to extend its reasoning chain.

**Sample Command:**
```bash
python -m inference.inference.generate_s1 \
    --model_path Qwen/Qwen3-4B-Instruct-2507 \
    --dataset_path datasets/hmmt_2025_nov.json \
    --reasoning_prompt_path prompts/s1_prompt.txt \
    --output_path outputs/hmmt_qwen3_4b_s1.json \
    --n 8 \
    --max_steps 12 \
    --tp_size 4
```

## Prompts

For FrontierScience, use special prompt variants (e.g., `reasoning_sci_prompt.txt`).

## Evaluation

Evaluation is performed using the `evaluation.ipynb` notebook. It uses specialized evaluators in `utils/evaluators.py` to extract and verify answers (e.g., checking LaTeX `\boxed{}` expressions), based on `math_verify`. Running this notebook requires Pebble, which enables multi-processing for `math_verify`. See `outputs/sample_outputs.json` for an example output file.

**Steps to evaluate:**
1. Open `evaluation.ipynb`.
2. Load your generated output file (e.g., `outputs/sample_outputs.json`).
3. Run the `evaluate_rollouts` function to compute accuracy across steps.
4. Use `compute_metrics` to see the mean accuracy at each reasoning step.

The evaluator supports parallel processing of math problems and handles timeouts for complex LaTeX parsing.
