from typing import List, Optional
import argparse
import os
import random
import re

from transformers import Trainer, PreTrainedTokenizer
import pandas as pd
from sim_student.model import get_base_model, get_model, get_tokenizer, generate_vllm
from sim_student.data_loading import load_train_val_data, load_test_data, process_kc_mastery
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.training_utils import TrainingCollator, get_training_args
from sim_student.utils import run_gc, initialize_seeds, merge_defaults
import wandb
from pdb import set_trace


PROFILE_SYSTEM_PROMPT = """You are a math education expert. Your task is to analyze a student's history interventions which include attempted questions correctness and dialogues between the student and tutor, then generate a student profile that captures the student's knowledge level on various knowledge components, learning patterns, and communication style."""
PROFILE_SYSTEM_PROMPT_NUM = """You are a math education expert. Your task is to analyze a student's history interventions which include attempted questions correctness and dialogues between the student and tutor, then generate a student profile that captures the student's knowledge level on various knowledge components, learning patterns, and communication style. The representation for all knowledge components in Knowledge State section should be in the format: 'KC: correct count/total attempted count'."""

MAX_LEN = 14500

def format_dialogue(dialogue: Dialogue):
    lines = []
    for idx, turn in enumerate(dialogue["turns"]):
        role = "Student" if turn["role"] == "student" else "Tutor"
        lines.append(f"Turn {idx + 1} ({role}): {turn['content']}")
    return "\n".join(lines)


def get_profile_prompt(tokenizer: PreTrainedTokenizer, dialogue: Dialogue, numerical=False):
    prompt = dialogue["history_context"].strip()
    system_prompt = PROFILE_SYSTEM_PROMPT_NUM if numerical else PROFILE_SYSTEM_PROMPT
    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


class ProfileSFTDataset(DatasetBase):
    def __init__(self, data: List[Dialogue], tokenizer: PreTrainedTokenizer, args: dict, drop_long_seqs: bool = False, generation=False):
        self.data = []
        excluded = 0
        self.len_ls = []
        check = []
        for dialogue_idx, dialogue in enumerate(data):
            prompt = get_profile_prompt(tokenizer, dialogue, numerical=args["ks_num"])
            self.len_ls.append(len(prompt))
            if drop_long_seqs and len(prompt) > 21000:
                excluded += 1
                check.append(len(prompt))
                continue
            if generation:
                self.data.append({
                    "dialogue_idx": dialogue_idx,
                    "studentID": dialogue["studentID"],
                    "timestamp": dialogue["timestamp"],
                    "QuestionId": dialogue["QuestionId"],
                    "InterventionId": dialogue["InterventionId"],
                    "IsCorrect": dialogue["IsCorrect"],
                    "prompt": prompt,
                })
            else:
                self.data.append({
                    "dialogue_idx": dialogue_idx,
                    "prompt": prompt,
                    "label": dialogue["profile_prev"].strip(),
                })

        print(f"Prompt length - mean: {sum(self.len_ls) / len(self.len_ls)}, max: {max(self.len_ls)}")
        print(f"Num dialogues: {len(self.data)} ({excluded} excluded labels/dialogues)")


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
    # train_data, val_data = process_kc_mastery(train_data), process_kc_mastery(val_data) if args["ks_num"] else (train_data, val_data)


    train_dataset = ProfileSFTDataset(train_data, tokenizer, args, drop_long_seqs=True)
    val_dataset = ProfileSFTDataset(val_data, tokenizer, args, drop_long_seqs=True)
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

    test_profile_agent(args)


def test_profile_agent(args: dict):
    test_data = load_test_data(args["dataset"])
    test_set = ProfileSFTDataset(test_data, get_tokenizer(args["base_model"]), args, drop_long_seqs=True)

    prompts = [sample["prompt"] for sample in test_set]
    predictions = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 500})

    test_df = pd.DataFrame(test_data)
    test_df['generated_profile'] = predictions

    save_path = f"data/annotated/eedi/test_{args['model_name']}_gen_profile.csv"
    test_df.to_csv(save_path, index=False)

    # generate_profile_iterative(args)

# Generate new profiles for train and val set for iterative training
def generate_profile_iterative(args: dict):
    train_data, val_data = load_train_val_data(args["dataset"])
    # train_data, val_data = process_kc_mastery(train_data), process_kc_mastery(val_data) if args["ks_num"] else (train_data, val_data)

    train_dataset = ProfileSFTDataset(train_data, get_tokenizer(args["base_model"]), args, drop_long_seqs=True)
    val_dataset = ProfileSFTDataset(val_data, get_tokenizer(args["base_model"]), args, drop_long_seqs=True)

    prompts = [sample["prompt"] for sample in train_dataset]
    predictions_train = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 500})

    prompts = [sample["prompt"] for sample in val_dataset]
    predictions_val = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 500})

    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    train_df['generated_profile'] = predictions_train
    val_df['generated_profile'] = predictions_val

    train_save_path = f"data/annotated/eedi/train_{args['model_name']}_gen_profile.csv"
    val_save_path = f"data/annotated/eedi/val_{args['model_name']}_gen_profile.csv"
    train_df.to_csv(train_save_path, index=False)
    val_df.to_csv(val_save_path, index=False)


def get_default_args(args: dict):
    if args["base_model"] == "3b":
        return {
            "train_batch_size": 2,
            "val_batch_size": 4,
            "grad_accum_steps": 32,
        }
    return {
        "train_batch_size": 1,
        "val_batch_size": 1,
        "grad_accum_steps": 64,
    }


def process_prompt(prompt):
    history_parts = prompt.split("Learning History ")[1]
    history_parts = history_parts.split("Decision Rules")[0]
    history_context = history_parts.strip()
    return 'Learning History ' + history_context


# Genrate student profile for QA side KT task
def generate_profile_for_QA(args: dict):
    df = pd.read_csv('data/annotated/eedi/QA_Profile/qa_gen_profile_total_dialogue.csv')
    set_trace()
    
    # valid_sub = df[~df['prompt'].isna()]
    # valid_sub['history_context'] = valid_sub['prompt'].apply(process_prompt)

    # subset = valid_sub.to_dict("records")
    # max_chunk_size = 10000

    # for chunk_idx in range(40000, len(subset), max_chunk_size):
    #     chunk = subset[chunk_idx : chunk_idx + max_chunk_size]
    #     test_set = ProfileSFTDataset(chunk, get_tokenizer(args["base_model"]), args, drop_long_seqs=True, generation=True)
    #     prompts = [sample["prompt"] for sample in test_set]

    #     predictions = generate_vllm(
    #         args["base_model"],
    #         prompts,
    #         args["model_name"],
    #         {"temperature": 0, "max_tokens": 500},
    #         batch=False,
    #     )

    #     res_df = pd.DataFrame(chunk)
    #     res_df["qa_generated_profile"] = predictions
    #     chunk_id = (chunk_idx // max_chunk_size) + 1
    #     chunk_path = f"data/annotated/eedi/qa_gen_profile_part_{chunk_id}.csv"
    #     res_df.to_csv(chunk_path, index=False)
    
    all_outputs = []
    for i in range(1, 8):
        chunk_path = f"data/annotated/eedi/qa_gen_profile_part_{i}.csv"
        chunk_df = pd.read_csv(chunk_path)
        all_outputs.append(chunk_df)

    set_trace()
    save_path = "data/annotated/eedi/QA_Profile/qa_gen_profile_total_dialogue.csv"
    combined = pd.concat(all_outputs, ignore_index=True)
    combined.to_csv(save_path, index=False)

def extract_n_question(history, n):
    item_blocks = re.findall(r"Item\s+\d+\s+\[Type:\s*.*?\]:.*?(?=\nItem\s+\d+\s+\[Type:|\Z)", history, flags=re.DOTALL)

    # Find indices of all Question blocks
    question_indices = [i for i, block in enumerate(item_blocks) if re.search(r"\[Type:\s*Question\s*\]", block)]
    # question_indices = [i for i, block in enumerate(item_blocks) if re.search(r"\[Type:\s*Tutoring Dialogue\s*\]", block)]

    # Keep only the last n Question indices
    kept_question_indices = set(question_indices[-n:]) if n > 0 else set()

    kept_blocks = []

    for i, block in enumerate(item_blocks):
        is_question = re.search(r"\[Type:\s*Question\s*\]", block)
        is_tutoring_dialogue = re.search(r"\[Type:\s*Tutoring Dialogue\s*\]", block)

        if is_tutoring_dialogue or i in kept_question_indices:
        # if is_question or i in kept_question_indices:
            kept_blocks.append(block)

    # Renumber all kept blocks from Item 1, Item 2, ...
    renumbered_blocks = []
    for new_item_num, block in enumerate(kept_blocks, start=1):
        block = re.sub(r"^Item\s+\d+", f"Item {new_item_num}", block)
        renumbered_blocks.append(block)

    return "\n\n".join(renumbered_blocks)


# Generate new profiles for test set for ablation
def generate_profile_context_ablation(args: dict):
    test_data = load_test_data(args["dataset"])
    test_data = pd.DataFrame(test_data)

    test_data['history_context'] = test_data['history_context'].apply(lambda x: extract_n_question(x, 0))
    test_data = test_data.to_dict("records")

    test_set = ProfileSFTDataset(test_data, get_tokenizer(args["base_model"]), args, drop_long_seqs=True)

    prompts = [sample["prompt"] for sample in test_set]
    predictions_test = generate_vllm(args["base_model"], prompts, args["model_name"], {"temperature": 0, "max_tokens": 500})

    test_df = pd.DataFrame(test_data)

    test_df['generated_profile'] = predictions_test

    test_save_path = f"data/annotated/eedi/test_{args['model_name']}_gen_profile.csv"
    test_df.to_csv(test_save_path, index=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--ks_num", action="store_true", help="Whether to use numerical knowledge state")
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
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

    generate_profile_context_ablation(args)

    # generate_profile_for_QA(args)

    # initialize_seeds(args["seed"])
    # train_profile_agent(args)


if __name__ == "__main__":
    main()
