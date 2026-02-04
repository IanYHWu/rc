# `RC` Training

This folder contains code for training with `RC`, based on verl (github.com/verl-project/verl) with FSDP. We modify some base verl functionalities, so you will need to install our version of the library. If you only need to do inference (`RC` decoding), you can omit installation of verl by using code from the `inference` folder instead.

## Environment setup
```bash
conda create -n verl python==3.10
conda activate verl
conda install nvidia/label/cuda-12.6.0::cuda-nvcc
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

git clone https://github.com/IanYHWu/rc
cd training
pip install -e .
pip install vllm==0.8.4
pip install flash-attn --no-build-isolation
pip install seaborn
pip install tensordict==0.6.2
pip install liger-kernel
```

## Usage
`RC`-specific code is contained inside `training/projects/reasoning_cache`. The important files are:
- `rollouts.py`: implements the main `RC` rollout generation logic.
- `summary_free_rollouts.py`: implements a version of `RC` that does not use summarization (for baselines).
- `ray_reasoning_cache_trainer.py`: implements the main training loop.
- `main_ppo_reasoning_cache.py`: main entry point for training.

The training configs are located in `training/projects/reasoning_cache/config`, and example scripts can be found in `training/projects/reasoning_cache/scripts`.
See `reasoning_cache_rl_script_flame.sh` for a script to launch (single-node) `RC` training, and `vanilla_flame.sh` for a script to launch standard GRPO.
You should be able to launch these scripts directly from command line. We used 1x8xH100 nodes for training.

The important `RC`-specific training parameters are:
- `reasoning_cache.online_rollout_steps`: number of training steps (`T_{train}`).
- `reasoning_cache.thinking_train_samples_per_online_rollout`:  number of sampled summaries for rollout generation (`N_{summ}`).
- `reasoning_cache.thinking_train_n_samples`:  number of parallel training rollouts (`K`).
- `reasoning_cache.use_replay_buffer`: whether to enable the summary replay buffer.
- `reasoning_cache.reasoning_prompt_path`:  reasoning prompt (`I_R`).
- `reasoning_cache.summarization_prompt_path`:  summarization prompt (`I_S`).
- `reasoning_cache.max_thinking_tokens`: max reasoning tokens (`H_R`).
- `reasoning_cache.max_summary_tokens`: max summarization tokens (`H_S`).
