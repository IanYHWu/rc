# Complete E2E

python -m reasoning_cache.inference.generate_e2e_complete \
--model_path HerrHruby/online_acemath_rl_30b_inst_even_harder_16k_filtered_3_steps_2_samples_step_70 \
--tp_size 4 \
--dataset_path reasoning_cache/datasets/hmmt_2025_nov.json \
--reasoning_prompt_path reasoning_cache/prompts/sum_e2e_complete_reasoning_prompt.txt \
--summarization_prompt_path reasoning_cache/prompts/sum_e2e_complete_summarization_prompt.txt \
--output_path /tmp/ianwu/hmmt_nov_online_acemath_rl_30b_inst_even_harder_16k_filtered_3_steps_2_samples_step_70.json \
--n 16 \
--max_steps 12 \
--temperature 1.0 \
--top_p 1.0 \
--max_thinking_tokens 16384 \
--tokenizer_path Qwen/Qwen3-30B-A3B-Instruct-2507 \
--model_class qwen \
--gpu_memory_utilization 0.8



# Baseline Generation


python -m reasoning_cache.inference.generate --model_type vllm --model_path Qwen/Qwen3-Next-80B-A3B-Instruct --dataset_path reasoning_cache/datasets/answerbench.json --output_path /tmp/ianwu/answerbench_qwen3_30b_next.json --prompt_path reasoning_cache/prompts/solution_generation_prompt.txt --n_rollouts_per_problem 4   --tp_size 4 --max_tokens 32768 --top_p 0.8 --temperature 0.7 --gpu_memory_utilization 0.8


# Self-Refine

python -m reasoning_cache.inference.generate_e2e_no_summary \
--model_path HerrHruby/offline_acemath_rl_4b_hard_with_dishsoap_16k_self_verify_step_80 \
--tp_size 8 \
--dataset_path reasoning_cache/datasets/frontierscience.json \
--reasoning_prompt_path reasoning_cache/prompts/self_refine_sci_prompt.txt \
--output_path /tmp/ianwu/fs_offline_acemath_rl_4b_hard_with_dishsoap_16k_self_verify_step_80.json \
--n 8 \
--max_steps 12 \
--temperature 1.0 \
--top_p 1.0 \
--max_thinking_tokens 16384

python -m reasoning_cache.inference.generate_e2e_no_summary \
--model_path HerrHruby/offline_acemath_rl_4b_inst_hard_with_dishsoap_16k_self_refine_step_70 \
--tp_size 8 \
--dataset_path reasoning_cache/datasets/frontierscience.json \
--reasoning_prompt_path reasoning_cache/prompts/self_correct_sci_prompt.txt \
--output_path /tmp/ianwu/fs_offline_acemath_rl_4b_inst_hard_with_dishsoap_16k_self_refine_step_70.json \
--n 8 \
--max_steps 12 \
--temperature 1.0 \
--top_p 1.0 \
--max_thinking_tokens 16384


# S1

python -m reasoning_cache.inference.generate_e2e_s1 \
--model_path Qwen/Qwen3-4B-Instruct-2507 \
--tp_size 4 \
--dataset_path reasoning_cache/datasets/aime.json \
--reasoning_prompt_path reasoning_cache/prompts/s1_prompt.txt \
--output_path /tmp/ianwu/aime_s1_base.json \
--n 1 \
--max_steps 12 \
--temperature 0.7 \
--top_p 0.8 \
--max_thinking_tokens 16384


