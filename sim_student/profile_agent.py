from typing import List, Optional
import argparse
import os
import random

from transformers import Trainer, PreTrainedTokenizer

from sim_student.model import get_base_model, get_model
from sim_student.data_loading import load_train_val_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.training_utils import TrainingCollator, get_training_args, MAX_LEN
from sim_student.utils import run_gc, initialize_seeds, merge_defaults
import wandb

PROFILE_SYSTEM_PROMPT = """You are a math education expert. Your task is to analyze a tutoring dialogue between a student and a tutor, and generate a student profile that captures the student's behavior, learning patterns, and communication style.

Use the dialogue to describe the student along these dimensions:
- Dialogue Acts: recurring patterns in how the student responds
- Correctness: how often the student answers accurately or shows uncertainty
- Error Patterns: notable mistakes or misconceptions that appear repeatedly
- Knowledge Acquisition: whether the student improves, stays stuck, or relies on the tutor
- Linguistic Style: tone, verbosity, confidence, and phrasing
- Interaction Style: how the student shapes the flow of the tutoring exchange

Be concise, pattern-focused, and grounded only in the dialogue. Do NOT reference specific turn numbers or positions, and describe patterns in aggregate instead of enumerating examples.

Output format:
Write one single paragraph that:
- Begins with concise analysis covering all dimensions above
- Ends with a 1-2 sentence overall summary"""


def format_dialogue(dialogue: Dialogue):
    lines = []
    for idx, turn in enumerate(dialogue["turns"]):
        role = "Student" if turn["role"] == "student" else "Tutor"
        lines.append(f"Turn {idx + 1} ({role}): {turn['content']}")
    return "\n".join(lines)


def get_profile_prompt(tokenizer: PreTrainedTokenizer, dialogue: Dialogue):
    prompt = "Dialogue:\n" + format_dialogue(dialogue)
    chat = [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


class ProfileSFTDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, drop_long_seqs: bool = False):
        self.data = []
        excluded = 0
        missing = 0
        for dialogue_idx, dialogue in enumerate(data):
            if not dialogue.get("turns") or not dialogue.get("gen_profile"):
                missing += 1
                continue
            prompt = get_profile_prompt(tokenizer, dialogue)
            if drop_long_seqs and len(prompt) > MAX_LEN:
                excluded += 1
                continue
            self.data.append({
                "dialogue_idx": dialogue_idx,
                "prompt": prompt,
                "label": dialogue["gen_profile"].strip(),
            })
        print(f"Num dialogues: {len(self.data)} ({excluded} excluded, {missing} missing labels/dialogues)")



def train_profile_agent(args: dict):
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(
        base_model,
        False,
        pt_model_name=args["pt_model_name"],
        r=args["r"],
        lora_alpha=args["lora_alpha"],
        quantize=args["quantize"],
    )

    # Load data
    train_data, val_data = load_train_val_data(args["dataset"])

    if args["subsample"]:
        train_start = int(len(train_data) * (1 - args["subsample"]))
        val_start = int(len(val_data) * (1 - args["subsample"]))
        train_data = train_data[train_start:]
        val_data = val_data[val_start:]

    train_dataset = ProfileSFTDataset(train_data, tokenizer, drop_long_seqs=True)
    val_dataset = ProfileSFTDataset(val_data, tokenizer, drop_long_seqs=True)
    collator = TrainingCollator(tokenizer)

    if args["wandb"]:
        wandb.login(key=args["wandb_key"], verify=True)
        wandb.init(project="profile_agent")
        wandb.config.update(args, allow_val_change=True)
        print("Run id:", wandb.run.id)

    trainer = Trainer(
        model=model,
        args=get_training_args(args),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
    del trainer, base_model, model
    run_gc()


def get_default_args(args: dict):
    if args["base_model"] == "3b":
        return {
            "train_batch_size": 2,
            "val_batch_size": 4,
            "grad_accum_steps": 32,
        }
    return {
        "train_batch_size": 1,
        "val_batch_size": 2,
        "grad_accum_steps": 64,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--subsample", type=float, default=0, help="Subsample dataset (0 for no subsampling), take from end of split")
    parser.add_argument("--seed", type=int, default=221)
    parser.add_argument("--train_batch_size", type=int, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--wandb", action="store_true", help="Whether to log training with wandb")
    parser.add_argument("--wandb_key", type=str, default="ff70920d9852a9d2e78bbd1cd2e100154d2c9c7d", help="API key for Weights & Biases.")
    args = parser.parse_args().__dict__
    args = merge_defaults(args, get_default_args(args))

    initialize_seeds(args["seed"])
    train_profile_agent(args)


if __name__ == "__main__":
    main()
