import os
import ray
import hydra

from projects.reasoning_cache.ray_reasoning_cache_trainer import (
    RayReasoningCacheTrainer,
)
from projects.reasoning_cache.rewards.rewards import RewardProcessor


@hydra.main(
    config_path="config",
    config_name="reasoning_cache_trainer",
    version_base=None,
)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                }
            }
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


def get_custom_reward_fn(config):
    import importlib.util, sys

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    function_name = reward_fn_config.get("name")
    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@ray.remote(num_cpus=1)
class TaskRunner:

    def run(self, config):
        from verl.utils.fs import copy_to_local

        # print initial config
        from pprint import pprint
        from omegaconf import OmegaConf

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_tokenizer, hf_processor

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        tokenizer.model_max_length = 131072
        processor = hf_processor(
            local_path, use_fast=True
        )  # used for multimodal LLM, could be none

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            from projects.reasoning_cache.ppo_workers import ActorRolloutRefWorker
            from verl.single_controller.ray import RayWorkerGroup

            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            raise NotImplementedError
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        if config.reward_model.enable:
            raise NotImplementedError("Using a reward model is not supported yet")

        # use reference model
        if (
            config.algorithm.use_kl_in_reward
            or config.actor_rollout_ref.actor.use_kl_loss
        ):
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        compute_score = get_custom_reward_fn(config)
        # reward_processor = RewardProcessor(compute_score, tokenizer, config)
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        trainer = RayReasoningCacheTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=compute_score,
            val_reward_fn=compute_score,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()

# Ugh
