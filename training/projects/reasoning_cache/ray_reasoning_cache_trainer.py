import uuid
from pprint import pprint
from typing import Dict, List, Optional, Any, Callable
import os
import json
from contextlib import contextmanager
from enum import Enum
import copy

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from codetiming import Timer

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    WorkerType,
    ResourcePoolManager,
    get_template,
    compute_response_mask,
    AdvantageEstimator,
)
from verl.trainer.ppo.metric_utils import (
    compute_throughput_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

from projects.reasoning_cache.rollouts import (
    ReasoningCacheRolloutGenerator,
)
from projects.reasoning_cache.summary_free_rollouts import (
    SummaryFreeRolloutGenerator,
)
from projects.reasoning_cache.advantage import compute_advantage
from projects.reasoning_cache.reasoning_cache_metrics import compute_data_metrics


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """
    Context manager for timing operations and storing results in a dictionary.

    Args:
        name: The name of the operation being timed.
        timing_raw: Dictionary to store timing results, where the operation name
                   will be used as the key and the elapsed time as the value.

    Yields:
        None: The context manager yields control to the wrapped code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def downsample_batch(batch: DataProto, world_size: int) -> Optional[DataProto]:
    """
    Downsamples a DataProto object to have a batch size divisible by world_size,
    but always keeps all samples with is_completion=True.
    """
    current_size = len(batch)
    if current_size < world_size:
        print(f"Batch size {current_size} < world_size {world_size}. Returning None.")
        return None

    target_size = (current_size // world_size) * world_size

    if target_size == current_size:
        return batch

    is_completion = np.array(batch.non_tensor_batch["is_completion"], dtype=bool)
    completion_indices = np.where(is_completion)[0]
    non_completion_indices = np.where(~is_completion)[0]

    if len(completion_indices) > target_size:
        raise ValueError("completion_indices should never exceed target_size.")

    remaining_slots = target_size - len(completion_indices)

    if remaining_slots < 0:
        raise ValueError("Not enough room to fit all completion samples into target size.")

    selected_non_completions = np.random.choice(
        non_completion_indices, 
        size=remaining_slots, 
        replace=False
    ) if remaining_slots > 0 else []

    selected_indices = np.sort(np.concatenate([completion_indices, selected_non_completions]))
    downsampled_batch = batch[selected_indices]

    return downsampled_batch


class RayReasoningCacheTrainer(RayPPOTrainer):
    """
    A Ray-based PPO trainer that implements reasoning cache training.
    """

    def __init__(
        self,
        config: OmegaConf,
        tokenizer: Any,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor: Optional[Any] = None,
        reward_fn: Optional[Callable] = None,
        val_reward_fn: Optional[Callable] = None,
        device_name: str = "cuda",
    ) -> None:
        """
        Initialize the RayReasoningCacheTrainer.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer for text processing.
            role_worker_mapping: Mapping of roles to worker types for distributed training.
            resource_pool_manager: Manager for GPU resource allocation.
            ray_worker_group_cls: Class for creating Ray worker groups.
            processor: Optional data processor for preprocessing.
            reward_fn: Function to compute rewards during training.
            val_reward_fn: Function to compute rewards during validation.
            device_name: Device to use for training (default: "cuda").
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.async_rollout_mode = False
        self.use_critic = False
        self.use_rm = False

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.device_name = device_name

        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError("KL in reward is not supported.")

        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise NotImplementedError("Only GRPO is supported.")

        self._validate_config()

        if (
            "return_raw_chat" not in self.config.data
            or self.config.data["return_raw_chat"] is False
        ):
            print(
                "Warning: setting return_raw_chat to True to enable training."
            )
        self.config.data["return_raw_chat"] = (
            True  # This is used in self._create_dataloader()
        )
        self._create_dataloader()

        with open(self.config.reasoning_cache.reasoning_prompt_path, "r") as f:
            self.reasoning_prompt_template = f.read()
        
        # Check if summary-free mode is enabled
        self.summary_free = self.config.reasoning_cache.get("summary_free", False)
        
        if self.summary_free:
            print("Summary-free mode enabled. Skipping summarization prompt template.")
            self.summarization_prompt_template = None
        else:
            with open(self.config.reasoning_cache.summarization_prompt_path, "r") as f:
                self.summarization_prompt_template = f.read()

        self.n = self.config.actor_rollout_ref.rollout.get("n", 2)
        self.train_batch_size = self.config.data.train_batch_size

    def _dump_generations(
        self,
        prompts: List[str],
        responses: List[str],
        dump_path: str,
        num_rollouts_to_dump: int,
    ) -> None:
        """
        Dump rollout/validation samples as JSONL format for analysis.

        This method saves training or validation generations to a JSONL file for
        later analysis. 

        Args:
            inputs: List of input prompts.
            outputs: List of generated responses.
            dump_path: Directory path where the JSONL file will be saved.
            num_rollouts_to_dump: Maximum number of rollouts to dump to the file.
        """
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}_train_rollouts.jsonl")

        n = min(num_rollouts_to_dump, len(prompts))
        base_data = {
            "prompt": prompts,
            "response": responses,
            "step": [self.global_steps] * n,
        }

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped {n} generations to {filename}")

    def _validate_config(self):
        config = self.config
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                         config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                         "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")
            
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean", "step-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_rollout_generator(self, sampling_params: Dict[str, Any]):
        """
        Create the appropriate rollout generator based on config.
        """
        if self.summary_free:
            return SummaryFreeRolloutGenerator(
                actor_rollout_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                reasoning_prompt_template=self.reasoning_prompt_template,
                reward_function=self.reward_fn,
                config=self.config,
                sampling_params=sampling_params
            )
        else:
            return ReasoningCacheRolloutGenerator(
                actor_rollout_wg=self.actor_rollout_wg,
                tokenizer=self.tokenizer,
                reasoning_prompt_template=self.reasoning_prompt_template,
                summarization_prompt_template=self.summarization_prompt_template,
                reward_function=self.reward_fn,
                config=self.config,
                sampling_params=sampling_params
            )

    def _validate(self) -> Dict[str, float]:
        """
        Perform validation on the model using the validation dataset.
        """
        rollout_generator = self._create_rollout_generator(
            sampling_params=self.config.actor_rollout_ref.rollout.val_kwargs
        )
        val_metrics = {}
        for batch_dict in self.val_dataloader:
            batch = DataProto.from_single_dict(batch_dict)
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt", "reward_model"],
            )
            uids = np.array(
                [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
            )
            gen_batch.non_tensor_batch["problem_ids"] = uids
            val_metrics.update(
                rollout_generator.generate_rollouts(
                    gen_batch,
                    eval_mode=True
                )[1]
            )

        return {f"val/{k}": v for k, v in val_metrics.items()}

    def _create_dataloader(self) -> None:
        """
        Create training and validation data loaders.
        """
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            from_hf_hub=self.config.data.get("from_hf_hub", True),
            hf_split=self.config.data.get("hf_split_train", "train"),
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            num_workers=self.config.data.get("filter_overlong_prompts_workers", None),
            chat_template_func=get_template(
                self.config.data.get("prompt_template", None)
            ),
        )
        assert self.train_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # Use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get(
                "gen_batch_size", self.config.data.train_batch_size
            ),
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            from_hf_hub=self.config.data.get("from_hf_hub", True),
            hf_split=self.config.data.get("hf_split_val", "test"),
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
            num_workers=self.config.data.get("filter_overlong_prompts_workers", None),
            chat_template_func=get_template(
                self.config.data.get("prompt_template", None)
            ),
        )
        assert self.val_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to actor optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )

    def fit(self) -> None:
        """
        Execute the main training loop.
        """

        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.global_compute_steps = 0

        # Load checkpoint before doing anything
        self._load_checkpoint()

        # Perform validation before training
        # Currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)

            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # We start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.dump_every = self.config.trainer.get("dump_every", 1)

        sampling_params = {
            "n": self.config.actor_rollout_ref.rollout.get("n", 2),
            "temperature": self.config.actor_rollout_ref.rollout.get("temperature", 0.6),
            "top_p": self.config.actor_rollout_ref.rollout.get("top_p", 1.0),
        }

        rollout_generator = self._create_rollout_generator(sampling_params=sampling_params)

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                print(f"global_steps: {self.global_steps}, fetching batch_dict")
                batch = DataProto.from_single_dict(batch_dict)

                # Pop keys for generation
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt", "reward_model"],
                )
                # Deal with uids
                uids = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                gen_batch.non_tensor_batch["problem_ids"] = uids

                is_last_step = self.global_steps >= self.total_training_steps
                with _timer("step", timing_raw):
                    # Generate a batch
                    with _timer("gen", timing_raw):
                        batch, online_rollout_metrics = (
                            rollout_generator.generate_rollouts(
                                gen_batch
                            )
                        )
                        online_rollout_metrics = {f"scores/{k}": v for k, v in online_rollout_metrics.items()}
                        metrics.update(online_rollout_metrics)
                    batch = downsample_batch(
                        batch, world_size=self.resource_pool_manager.get_n_gpus())

                    # Balance the batch by token count
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # Compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch['response_mask'] = compute_response_mask(batch)

                    with _timer("adv", timing_raw):
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch)                    

                    with _timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()
                        if self.config.reasoning_cache.get("use_replay_buffer", True):
                            save_path = os.path.join(self.config.trainer.default_local_dir, f"replay_buffer_global_step_{self.global_steps}")
                            rollout_generator.save_replay_buffer(save_path)

                # Collect metrics
                metrics.update(
                    compute_data_metrics(
                        batch, 
                        self.config
                    )
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                metrics.update(
                    {
                        "steps/steps": self.global_steps,
                        "steps/compute_matched_steps": self.global_compute_steps,
                    }
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughput_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                logger.log(data=metrics, step=self.global_steps)

                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                num_rollouts_to_dump = self.config.trainer.get(
                    "num_rollouts_to_dump", 1
                )
                if (
                    rollout_data_dir
                    and self.dump_every > 0
                    and ((self.global_steps - 1) % self.dump_every == 0)
                ):
                    with _timer("dump_rollout_generations", timing_raw):
                        n_rollouts = len(batch.batch["prompts"])
                        sampled_indices = np.random.choice(
                            [i for i in range(n_rollouts)],
                            size=num_rollouts_to_dump,
                            replace=False,
                        )
                        inputs = self.tokenizer.batch_decode(
                            batch.batch["prompts"][sampled_indices],
                            skip_special_tokens=True,
                        )
                        outputs = self.tokenizer.batch_decode(
                            batch.batch["responses"][sampled_indices],
                            skip_special_tokens=True,
                        )
                        self._dump_generations(
                            prompts=inputs,
                            responses=outputs,
                            dump_path=rollout_data_dir,
                            num_rollouts_to_dump=num_rollouts_to_dump,
                        )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.global_compute_steps += (
                    torch.sum(batch.batch["response_mask"]).detach().item()
                )
