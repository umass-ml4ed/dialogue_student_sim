import argparse
import os
import warnings
from typing import Any
import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset
from peft import PeftModel
from transformers import PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
import pandas as pd

from sim_student.data_loading import load_train_val_data, load_test_data
from sim_student.model import get_base_model, get_model
from sim_student.profile_agent import get_profile_prompt, test_profile_agent
from sim_student.prompting import get_local_prompt
from sim_student.utils import get_checkpoint_path, initialize_seeds, merge_defaults, run_gc
from sim_student.testing import test
from pdb import set_trace

BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007

PROMPT_MAX_LEN = 1700

def build_profile_grpo_dataset(data: list[dict[str, Any]], tokenizer: PreTrainedTokenizer, args: dict) -> HFDataset:
    rows = []
    excluded = 0
    token_len_ls = []
    res_len_ls = []

    for row in data:
        prompt = get_profile_prompt(tokenizer, row, numerical=args["ks_num"])
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        token_len_ls.append(prompt_len)
        if prompt_len > PROMPT_MAX_LEN:
            excluded += 1
            continue

        # output = row.get("profile_prev", "").strip()
        # output_len = len(tokenizer(output, add_special_tokens=False)["input_ids"])
        # res_len_ls.append(output_len)

        valid_keys = ['studentID', 'InterventionId', 'turns', 'question', 'CorrectAnswer', 'profile_prev', 'history_context']
        parsed_row = {key: row.get(key) for key in valid_keys}
        parsed_row["prompt"] = prompt
        rows.append(parsed_row)

    print(f"Num dialogues: {len(rows)} ({excluded} excluded for prompt length)")

    return HFDataset.from_list(rows)


def mask_non_target_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    for idx in range(len(labels)):
        boh_idxs = (labels[idx] == BOH_TOKEN_ID).nonzero()
        eoh_idxs = (labels[idx] == EOH_TOKEN_ID).nonzero()
        labels[idx, :eoh_idxs[2] + 1] = -100 # Mask labels up to end of first assistant header
        for header_ct in range(3, len(boh_idxs), 2):
            # Mask labels between start of each user header to end of subsequent assistant header
            end_idx = eoh_idxs[header_ct + 1] if header_ct + 1 < len(eoh_idxs) else labels.shape[1] # In case dialogue ends with a user turn
            labels[idx, boh_idxs[header_ct] : end_idx + 1] = -100

    return labels


class DialogueLikelihoodReward:
    def __init__(
        self,
        model: PeftModel,
        tokenizer: PreTrainedTokenizer,
        role: str,
        reward_batch_size: int,
        reward_adapter_name: str = "reward",
        policy_adapter_name: str = "default",
        reduction: str = "sum",
    ):
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"Unsupported dialogue reward reduction: {reduction}")

        self.model = model
        self.tokenizer = tokenizer
        self.role = role
        self.reward_batch_size = reward_batch_size
        self.reward_adapter_name = reward_adapter_name
        self.policy_adapter_name = policy_adapter_name
        self.reduction = reduction


    def _score_batch(self, examples: list[dict[str, Any]]) -> list[float]:
        prompts = [
            get_local_prompt(
                dialogue=example,
                role=self.role,
                tokenizer=self.tokenizer,
                input_type="profile",
                numerical=True,
            )
            for example in examples
        ]

        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)

        device = next(self.model.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        labels = mask_non_target_tokens(input_ids, attention_mask).to(device)

        # switch to reward adapter and compute log likelihoods for reward
        previous_training_mode = self.model.training
        self.model.set_adapter(self.reward_adapter_name)
        self.model.eval()

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # switch back to policy adapter
        self.model.set_adapter(self.policy_adapter_name)
        self.model.train(previous_training_mode)
        self.tokenizer.padding_side = original_padding_side

        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs.masked_fill(~valid_mask, 0.0)

        if self.reduction == "sum":
            rewards = token_logprobs.sum(dim=1)
        else:
            token_counts = valid_mask.sum(dim=1).clamp_min(1)
            rewards = token_logprobs.sum(dim=1) / token_counts

        return rewards.detach().cpu().tolist()

    def __call__(
        self,
        completions: list[str] | None = None,
        turns: list[Any] | None = None,
        log_metric=None,
        **kwargs: Any,
    ) -> list[float]:
        if not completions:
            return []

        rewards = []
        batch_examples = []
        completion_count = len(completions)

        for idx, profile_text in enumerate(completions):
            row = {'question':kwargs['question'][idx], 'profile_prev': profile_text.strip(), 'turns': turns[idx]}

            batch_examples.append(row)

            if len(batch_examples) == self.reward_batch_size:
                rewards.extend(self._score_batch(batch_examples))
                batch_examples = []

        if batch_examples:
            rewards.extend(self._score_batch(batch_examples))

        if log_metric and rewards:
            log_metric("reward/dialogue_log_likelihood_mean", sum(rewards) / len(rewards))

        return rewards

    @property
    def __name__(self):
        return f"dialogue_log_likelihood_{self.reduction}_reward"


def get_grpo_args(args: dict) -> GRPOConfig:
    return GRPOConfig(
        output_dir=get_checkpoint_path(args["model_name"]),
        num_train_epochs=args["epochs"],
        learning_rate=args["lr"],
        weight_decay=args["wd"],
        max_grad_norm=args["gc"] or None,
        per_device_train_batch_size=args["train_batch_size"],
        per_device_eval_batch_size=args["val_batch_size"],
        gradient_accumulation_steps=args["grad_accum_steps"],
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        remove_unused_columns=False,
        report_to="wandb" if args["wandb"] else "none",
        logging_steps=args["logging_steps"],
        bf16=not args["quantize"],
        gradient_checkpointing=args["gradient_checkpointing"],
        num_generations=args["num_generations"],
        num_generations_eval=args["num_generations_eval"],
        max_completion_length=args["max_completion_length"],
        temperature=args["temperature"],
        top_k=args["top_k"],
        beta=args["beta"],
        loss_type=args["loss_type"],
    )


def load_shared_model_with_policy_and_reward_adapters(args: dict) -> tuple[PeftModel, PreTrainedTokenizer]:
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])

    model = get_model(
        base_model,
        False,
        pt_model_name=args["pt_model_name"],
        r=args["r"],
        lora_alpha=args["lora_alpha"],
        quantize=args["quantize"],
        use_gradient_checkpointing=args["gradient_checkpointing"],
    )

    if not isinstance(model, PeftModel):
        raise TypeError("Expected a PeftModel for adapter switching.")

    if args["gradient_checkpointing"] and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    reward_adapter_path = get_checkpoint_path(args["dialogue_model_name"])
    model.load_adapter(reward_adapter_path, adapter_name="reward", is_trainable=False)
    model.set_adapter("default")

    return model, tokenizer


def train_dialogue_grpo(args: dict):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = world_size * args["train_batch_size"] * args["grad_accum_steps"]
    if effective_batch_size % args["num_generations"] != 0:
        raise ValueError(
            "GRPO requires the effective batch size "
            f"({effective_batch_size} = world_size * train_batch_size * grad_accum_steps) "
            f"to be divisible by num_generations ({args['num_generations']})."
        )
    
    model, tokenizer = load_shared_model_with_policy_and_reward_adapters(args)
    tokenizer.padding_side = "left"

    train_data, val_data = load_train_val_data(args["dataset"])
    train_dataset = build_profile_grpo_dataset(train_data, tokenizer, args)
    val_dataset = build_profile_grpo_dataset(val_data, tokenizer, args)

    reward_func = DialogueLikelihoodReward(
        model=model,
        tokenizer=tokenizer,
        role=args["dialogue_role"],
        reward_batch_size=args["dialogue_reward_batch_size"],
        reward_adapter_name="reward",
        policy_adapter_name="default",
        reduction=args["dialogue_reward_reduction"],
    )

    if args["wandb"]:
        wandb.login(key=args["wandb_key"], verify=True)
        wandb.init(project="profile_grpo")
        wandb.config.update(args, allow_val_change=True)
        print("Run id:", wandb.run.id)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=get_grpo_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model()

    del trainer, model
    run_gc()

    test_profile_agent(args)

    # student_model should be set to dialogue_model_name for dialogue performance evaluation after DPO on new profile is trained
    test({
        **args,
        "temperature": 0.0,
        # **({"student_model": 'eedi-stud-sft-8b_profile'})
        **({"student_model": args["dialogue_model_name"]})
    })


def get_default_args(args: dict):
    if args["base_model"] == "3b":
        return {
            "train_batch_size": 2,
            "val_batch_size": 4,
            "grad_accum_steps": 32,
            "num_generations": 4,
            "num_generations_eval": 4,
            "dialogue_reward_batch_size": 2,
        }
    return {
        "train_batch_size": 1,
        "val_batch_size": 2,
        "grad_accum_steps": 64,
        "num_generations": 4,
        "num_generations_eval": 2,
        "dialogue_reward_batch_size": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--role", choices=["student", "tutor"], default="student")
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
    parser.add_argument("--input_type", choices=["none", "profile", "dialogue"], default="none")
    parser.add_argument("--ks_num", action="store_true", help="Whether to use numerical knowledge state")
    parser.add_argument("--iterative", action="store_true", help="Whether to use iterative annotation")
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--subsample", type=float, default=0)
    parser.add_argument("--seed", type=int, default=221)
    parser.add_argument("--train_batch_size", type=int, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--max_completion_length", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable activation checkpointing to reduce GPU memory usage.")
    parser.add_argument("--num_generations", type=int, help="Rollouts sampled per prompt during training")
    parser.add_argument("--num_generations_eval", type=int, help="Rollouts sampled per prompt during evaluation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", default="dapo")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--dialogue_model_name", required=True, help="Frozen dialogue model adapter/checkpoint for reward computation.")
    parser.add_argument("--dialogue_quantize", action="store_true", help="Must match --quantize in adapter-switch mode.")
    parser.add_argument("--dialogue_role", choices=["student", "tutor"], default="student")
    parser.add_argument("--dialogue_reward_batch_size", type=int, help="Batch size for frozen dialogue-model reward computation.")
    parser.add_argument("--dialogue_reward_reduction", choices=["sum", "mean"], default="mean")
    parser.add_argument("--wandb", action="store_true", help="Whether to log training with wandb")
    parser.add_argument("--wandb_key", type=str, default="ff70920d9852a9d2e78bbd1cd2e100154d2c9c7d")

    args = parser.parse_args().__dict__

    args = merge_defaults(args, get_default_args(args))
    initialize_seeds(args["seed"])
    train_dialogue_grpo(args)

    # test_profile_agent(args)


if __name__ == "__main__":
    main()
