import argparse
import os
from ast import literal_eval
from typing import Any

import torch
import torch.nn.functional as F
import wandb
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer

from sim_student.data_loading import load_train_val_data
from sim_student.model import get_base_model, get_model
from sim_student.profile_agent import get_profile_prompt
from sim_student.prompting import get_local_prompt
from sim_student.training_utils import MAX_LEN
from sim_student.utils import get_checkpoint_path, initialize_seeds, merge_defaults, run_gc

BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        return "".join(_completion_to_text(item) for item in completion)
    return str(completion)


def build_profile_grpo_dataset(data: list[dict[str, Any]], tokenizer: PreTrainedTokenizer, max_prompt_length: int | None) -> HFDataset:
    rows = []
    excluded = 0
    prompt_limit = max_prompt_length or MAX_LEN

    for dialogue_idx, row in enumerate(data):
        prev_turns = row['prev_turns']
        current_turns = row['turns']

        prompt = get_profile_prompt(tokenizer, {"turns": prev_turns})
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if prompt_len > prompt_limit:
            excluded += 1
            continue

        parsed_row = {key: value for key, value in row.items()}
        parsed_row["dialogue_idx"] = dialogue_idx
        parsed_row["prompt"] = prompt
        rows.append(parsed_row)

    print(
        f"Num dialogues: {len(rows)} "
        f"({excluded} excluded for prompt length)"
    )
    return HFDataset.from_list(rows)


def _mask_non_target_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor, role: str) -> torch.Tensor:
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    target_header_idx = 2 if role == "student" else 1
    for idx in range(len(labels)):
        boh_idxs = (labels[idx] == BOH_TOKEN_ID).nonzero(as_tuple=False).flatten()
        eoh_idxs = (labels[idx] == EOH_TOKEN_ID).nonzero(as_tuple=False).flatten()

        if len(eoh_idxs) <= target_header_idx:
            labels[idx] = -100
            continue

        labels[idx, : eoh_idxs[target_header_idx] + 1] = -100
        for header_ct in range(target_header_idx + 1, len(boh_idxs), 2):
            end_idx = eoh_idxs[header_ct + 1] if header_ct + 1 < len(eoh_idxs) else labels.shape[1]
            labels[idx, boh_idxs[header_ct] : end_idx + 1] = -100

    return labels


class DialogueLikelihoodReward:
    def __init__(self, model: torch.nn.Module, tokenizer: PreTrainedTokenizer, role: str, reward_batch_size: int, reduction: str = "sum"):
        if reduction not in {"sum", "mean"}:
            raise ValueError(f"Unsupported dialogue reward reduction: {reduction}")

        self.model = model.eval()
        self.tokenizer = tokenizer
        self.role = role
        self.reward_batch_size = reward_batch_size
        self.reduction = reduction

    def _build_dialogue_example(self, row: dict[str, Any], profile_text: str, prev_turns: list[dict[str, Any]], turns: list[dict[str, Any]]) -> dict[str, Any]:
        example = dict(row)
        example["profile_prev"] = profile_text.strip()
        example["turn_prev"] = prev_turns
        example["turns"] = turns
        return example

    def _score_batch(self, examples: list[dict[str, Any]]) -> list[float]:
        prompts = [
            get_local_prompt(
                dialogue=example,
                role=self.role,
                tokenizer=self.tokenizer,
                input_type="profile",
            )
            for example in examples
        ]

        self.tokenizer.padding_side = "right"
        tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        device = next(self.model.parameters()).device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        labels = _mask_non_target_tokens(input_ids, attention_mask, self.role)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        valid_mask = shift_labels != -100
        safe_labels = shift_labels.masked_fill(~valid_mask, 0)

        token_logprobs = F.log_softmax(shift_logits, dim=-1).gather(
            dim=-1,
            index=safe_labels.unsqueeze(-1),
        ).squeeze(-1)
        token_logprobs = token_logprobs.masked_fill(~valid_mask, 0.0)

        if self.reduction == "sum":
            rewards = token_logprobs.sum(dim=1)
        else:
            token_counts = valid_mask.sum(dim=1).clamp_min(1)
            rewards = token_logprobs.sum(dim=1) / token_counts

        return rewards.detach().cpu().tolist()

    def __call__(self, completions: list[str] | None = None, prev_turns: list[Any] | None = None, turns: list[Any] | None = None, log_metric=None, **kwargs: Any) -> list[float]:
        if completions is None or prev_turns is None or turns is None:
            raise ValueError("DialogueLikelihoodReward requires 'completions', 'prev_turns', and 'turns'.")

        rewards = []
        batch_examples = []
        completion_count = len(completions)

        for idx, profile_text in enumerate(completions):
            row = {}
            for key, values in kwargs.items():
                if isinstance(values, list) and len(values) == completion_count:
                    row[key] = _maybe_parse(values[idx])

            batch_examples.append(
                self._build_dialogue_example(
                    row=row,
                    profile_text=_completion_to_text(profile_text),
                    prev_turns=_maybe_parse(prev_turns[idx]),
                    turns=_maybe_parse(turns[idx]),
                )
            )

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


def load_dialogue_reward_model(args: dict):
    reward_base_model, reward_tokenizer = get_base_model(args["dialogue_base_model"], args["dialogue_quantize"])
    reward_model = get_model(reward_base_model, True, model_name=args["dialogue_model_name"])
    reward_model.eval()
    return reward_model, reward_tokenizer


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
        num_generations=args["num_generations"],
        num_generations_eval=args["num_generations_eval"],
        max_prompt_length=args["max_prompt_length"],
        max_completion_length=args["max_completion_length"],
        temperature=args["temperature"],
        top_p=args["top_p"],
        top_k=args["top_k"],
        min_p=args["min_p"],
        beta=args["beta"],
        epsilon=args["epsilon"],
        loss_type=args["loss_type"],
        scale_rewards=args["scale_rewards"],
        log_completions=args["log_completions"],
    )


def train_dialogue_grpo(args: dict):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = world_size * args["train_batch_size"] * args["grad_accum_steps"]
    if effective_batch_size % args["num_generations"] != 0:
        raise ValueError(
            "GRPO requires the effective batch size "
            f"({effective_batch_size} = world_size * train_batch_size * grad_accum_steps) "
            f"to be divisible by num_generations ({args['num_generations']})."
        )

    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    tokenizer.padding_side = "left"
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])

    reward_model, reward_tokenizer = load_dialogue_reward_model(args)
    reward_func = DialogueLikelihoodReward(
        model=reward_model,
        tokenizer=reward_tokenizer,
        role=args["dialogue_role"],
        reward_batch_size=args["dialogue_reward_batch_size"],
        reduction=args["dialogue_reward_reduction"],
    )

    train_data, val_data = load_train_val_data(args["dataset"])

    train_dataset = build_profile_grpo_dataset(train_data, tokenizer, args["max_prompt_length"])
    val_dataset = build_profile_grpo_dataset(val_data, tokenizer, args["max_prompt_length"])

    if args["wandb"]:
        wandb.login(key=args["wandb_key"], verify=True)
        wandb.init(project="profile_agent_grpo")
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

    del trainer, reward_model, reward_tokenizer, base_model, model
    run_gc()


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
        "num_generations_eval": 4,
        "dialogue_reward_batch_size": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eedi")
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
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--max_prompt_length", type=int, default=MAX_LEN)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, help="Rollouts sampled per prompt during training")
    parser.add_argument("--num_generations_eval", type=int, help="Rollouts sampled per prompt during evaluation")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--min_p", type=float)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--loss_type", default="dapo")
    parser.add_argument("--scale_rewards", default=False)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--dialogue_model_name", required=True, help="Frozen dialogue model adapter/checkpoint for reward computation.")
    parser.add_argument("--dialogue_base_model", help="Base model for the frozen dialogue model. Defaults to --base_model.")
    parser.add_argument("--dialogue_quantize", action="store_true", help="Load the frozen dialogue reward model in 4-bit.")
    parser.add_argument("--dialogue_role", choices=["student", "tutor"], default="student")
    parser.add_argument("--dialogue_reward_batch_size", type=int, help="Batch size for frozen dialogue-model reward computation.")
    parser.add_argument("--dialogue_reward_reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--wandb", action="store_true", help="Whether to log training with wandb")
    parser.add_argument("--wandb_key", type=str, default="ff70920d9852a9d2e78bbd1cd2e100154d2c9c7d", help="API key for Weights & Biases.")
    args = parser.parse_args().__dict__
    if args["dialogue_base_model"] is None:
        args["dialogue_base_model"] = args["base_model"]
    args = merge_defaults(args, get_default_args(args))

    initialize_seeds(args["seed"])
    train_dialogue_grpo(args)


if __name__ == "__main__":
    main()
