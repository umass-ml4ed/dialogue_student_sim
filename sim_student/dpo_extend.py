import argparse
from typing import List, Dict, Any
import random
from itertools import combinations
from scipy.stats import norm
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import datasets
from datasets import Dataset as HFDataset
from trl import DPOTrainer, DPOConfig
import pandas as pd
import numpy as np
import torch

from sim_student.data_loading import load_train_val_data
from sim_student.data_utils import Dialogue, DatasetBase
from sim_student.model import get_base_model, get_model, generate_vllm, get_tokenizer
from sim_student.prompting import *
from sim_student.training_utils import MAX_LEN
from sim_student.testing import test, TestingDataset
from sim_student.eval import eval_similarity_text, eval_similarity_acts, eval_knowledge_state, eval_similarity_correctness_and_errors, eval_tutor_ppl
from sim_student.utils import get_checkpoint_path, run_gc, initialize_seeds
import wandb

BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007

def get_local_prompt_dpo(dialogue: Dialogue, role: str, kcs_src: str = None, ending_turn: int = None, last_turn_sub: str = None, persona_type: str = None, input_type: str = None):
    assert role in ("student", "tutor")
    turns = [*dialogue["turns"]] # Copy to not modify original dialogue

    # Add special end of dialogue tag to end of last turn (except when doing full dialogue generation, i.e. done flag is present)
    if "done" not in dialogue:
        turns[-1] = {**turns[-1], "content": turns[-1]["content"] + END_OF_DIALOGUE}
    _ending_turn = ending_turn if ending_turn is not None else len(turns)
    if last_turn_sub: # Sub in alternative text for last turn, copy to not modify original dialogue
        turns[_ending_turn - 1] = {**turns[_ending_turn - 1], "content": last_turn_sub}

    # Create prompt context: profile/dialogue only input.
    context = ""
    if input_type != "none":
        if input_type == 'profile':
            input_context = dialogue["profile_prev"].replace("\n\n", " ").strip()
            context += f"Student Profile:\n{input_context}\n"
        else:
            input_context = ""
            for turn in dialogue["turn_prev"]:
                role_title = "Student" if turn["role"] == "student" else "Tutor"
                input_context += f"{role_title}: {turn['content']}\n"
            
            context += f"Previous Dialogue:\n{input_context}\n"


    if role == "student":
        context += get_formatted_persona(dialogue, persona_type, turn_idx=_ending_turn - 1)

    question = format_question(dialogue, kcs_src=kcs_src)

    # Question and dialogue are generated in completion.
    turns = turns[:_ending_turn]

    system_prompt = STUDENT_SYS_PROMPT_SHORT if role == "student" else TUTOR_SYS_PROMPT_SHORT

    if input_type != "none":
        if input_type == 'profile':
            system_prompt = STUDENT_SYS_PROMPT_SHORT_PROFILE
        else:
            system_prompt = STUDENT_SYS_PROMPT_SHORT_DIALOGUE

    return {
        "system_prompt": system_prompt,
        "context": context,
        "question": question,
        "remaining_turns": turns,
    }

def map_turns_to_chat_roles(turns, target_role= "student"):
    return [{
            "role": "assistant" if turn["role"] == target_role else "user",
            "content": turn["content"],
        } for turn in turns]


def build_prompt_messages(dialogue: Dialogue, kcs_src: str = None, persona_type: str = None, input_type: str = None, target_role: str = "student"):
    parts = get_local_prompt_dpo(dialogue=dialogue, role=target_role, kcs_src=kcs_src, persona_type=persona_type, input_type=input_type)
    if input_type == "none":
        return [{"role": "system", "content": parts["system_prompt"]}]

    return [{"role": "system", "content": parts["system_prompt"]},
            {"role": "user", "content": parts["context"]}]

# Question is included in completion to build more pairs for DPO training. In furture dataset with more data points per problem, can consider moving question to prompt and only include dialogue turns in completion.
def build_completion_messages(dialogue: Dialogue, kcs_src: str = None, persona_type: str = None, input_type: str = None, target_role: str = "student"):
    parts = get_local_prompt_dpo(dialogue=dialogue, role=target_role, kcs_src=kcs_src, persona_type=persona_type, input_type=input_type)

    # Keep question in completion, but start completion with assistant to preserve
    # a valid chat continuation boundary from prompt (which ends with user).
    completion_messages = [{"role": "assistant", "content": f"Question:\n{parts['question']}"}]
    completion_messages.extend(map_turns_to_chat_roles(parts["remaining_turns"], target_role=target_role))

    return completion_messages


# two ways to construct the (chosen, reject) pairs for DPO:
# 1) randomly select negative pair from other students
# 2) select negative pair based on generated profile patterns (patterns on act, knowledge, summary)

def construct_dpo_data(data: List[Dialogue], args: dict):
    def _profile_pattern_text(profile_text: str) -> str:
        profile_parts = [p.strip() for p in profile_text.split(". ") if p.strip()]

        if len(profile_parts) == 1:
            text = profile_parts[0]
        else:
            text = ". ".join(profile_parts[:2])

        if not text.endswith("."):
            text += "."
        return text

    df = pd.DataFrame(data)
    if args["random_form"]:
        examples = []
        for idx, dialogue in df.iterrows():
            # print(dialogue["studentID"])

            student_i = dialogue["studentID"]
            problem_i = dialogue["QuestionId"]

            select_pool = df[df["studentID"] != student_i]

            if len(select_pool) == 0:
                continue

            sampleN = min(args["num_pairs"], len(select_pool))
            random_sel = select_pool.sample(n=sampleN, replace=True)

            prompt_messages = build_prompt_messages(dialogue=dialogue, input_type=args.get("input_type"))

            chosen_messages = build_completion_messages(dialogue=dialogue, input_type=args.get("input_type"), target_role="student")

            for i in range(len(random_sel)):
                rejected_messages = build_completion_messages(random_sel.iloc[i], input_type=args.get("input_type"), target_role="student")

                examples.append({
                    "prompt": prompt_messages,
                    "chosen": chosen_messages,
                    "rejected": rejected_messages,
                })


        return HFDataset.from_list(examples)
    
    else:
        # use profile, first two sentences in generated profile is about Dialogue Acts and Correctness patterns.
        # Build (chosen, rejected) pairs by selecting other-student rows with cosine distance above threshold.
        examples = []

        dialogues = df.to_dict("records")
        n_rows = len(dialogues)

        # Build profile text used for distance comparisons (first two sentences only).
        profile_texts = [_profile_pattern_text(p) for p in df["profile_prev"].tolist()]

        # Encode all profile snippets once for efficiency.
        emb_batch_size = args.get("emb_batch_size", 16)
        embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B")
        embeddings = embedding_model.encode(profile_texts, batch_size=emb_batch_size, show_progress_bar=True, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)

        distance_threshold = args.get("epsilon", 0.3)
        max_pairs = args.get("num_pairs", 3)

        distance_matrix = 1.0 - (embeddings @ embeddings.T)

        # Select negatives from the same problem but different student.
        # question_ids = df["QuestionId"].to_numpy()
        student_ids = df["studentID"].to_numpy()
        # same_problem_mask = question_ids[:, None] == question_ids[None, :]
        other_student_mask = student_ids[:, None] != student_ids[None, :]
        hard_negative_mask = distance_matrix > distance_threshold
        valid_mask = other_student_mask & hard_negative_mask

        # Keep valid distances, mask invalid as -inf so they sink in descending sort.
        masked_distances = np.where(valid_mask, distance_matrix, -np.inf)

        k = min(max_pairs, n_rows - 1)
        sorted_candidate_idx = np.argsort(masked_distances, axis=1)[:, ::-1]
        topk_idx = sorted_candidate_idx[:, :k]
        topk_scores = np.take_along_axis(masked_distances, topk_idx, axis=1)
        topk_valid = np.isfinite(topk_scores)

        # Flatten valid (anchor_idx, negative_idx) pairs.
        anchor_idx = np.repeat(np.arange(n_rows), k)
        negative_idx = topk_idx.reshape(-1)
        pair_valid = topk_valid.reshape(-1)

        anchor_idx = anchor_idx[pair_valid]
        negative_idx = negative_idx[pair_valid]

        prompt_cache = [build_prompt_messages(dialogue=d, input_type=args.get("input_type")) for d in dialogues]
        chosen_cache = [build_completion_messages(dialogue=d, input_type=args.get("input_type"), target_role="student") for d in dialogues]
        rejected_cache = [build_completion_messages(dialogue=d, input_type=args.get("input_type"), target_role="student") for d in dialogues]

        examples = [{
                "prompt": prompt_cache[i],
                "chosen": chosen_cache[i],
                "rejected": rejected_cache[j],
            } for i, j in zip(anchor_idx.tolist(), negative_idx.tolist())]

        del embedding_model, embeddings
        run_gc()

        return HFDataset.from_list(examples)


            
# copying pad() from trl.trainer.utils due to import lead to much longer loading time, equivalent to "from trl.trainer.utils import pad"
def pad(tensors: list[torch.Tensor], padding_value: int = 0, padding_side: str = "right") -> torch.Tensor:
    output_shape = np.max([t.shape for t in tensors], 0).tolist()

    # Create an output tensor filled with the padding value
    output = torch.full((len(tensors), *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output



class StudentCompletionMaskCollator:
    def __init__(self, tokenizer, pad_token_id):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
    
    def _build_student_completion_mask(self, completion_ids):
        mask = [0] * len(completion_ids)
        if len(completion_ids) == 0:
            return mask

        ids = torch.tensor(completion_ids, dtype=torch.long)

        boh_idxs = (ids == BOH_TOKEN_ID).nonzero(as_tuple=False).flatten().tolist()
        eoh_idxs = (ids == EOH_TOKEN_ID).nonzero(as_tuple=False).flatten().tolist()

        # If there are no headers in the completion, treat the whole completion as student.
        if len(boh_idxs) == 0:
            return [1] * len(completion_ids)

        # Mark only assistant spans (student turns) after each header.
        num_headers = min(len(boh_idxs), len(eoh_idxs))
        for h in range(num_headers):
            header_start = boh_idxs[h] + 1
            header_end = eoh_idxs[h]
            header_role = self.tokenizer.decode(ids[header_start:header_end], skip_special_tokens=False).strip().lower()

            content_start = eoh_idxs[h] + 1
            content_end = boh_idxs[h + 1] if h + 1 < len(boh_idxs) else len(completion_ids)

            # content_text = self.tokenizer.decode(ids[content_start:content_end], skip_special_tokens=False).lstrip()
            if header_role == "assistant":
                for i in range(content_start, content_end):
                    mask[i] = 1

        return mask


    # attention mask is 1 for all non-padding tokens in prompt and completion, 0 for padding, completion mask is 1 for student tokens in completion
    def _build_sequence(self, prompt_ids: List[int], completion_ids: List[int]) -> Dict[str, torch.Tensor]:
        completion_mask = self._build_student_completion_mask(completion_ids)

        input_ids = prompt_ids + completion_ids
        attention_mask = [1] * len(input_ids)

        # single mask: 0 on prompt, student-only on completion
        full_completion_mask = [0] * len(prompt_ids) + completion_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "completion_mask": torch.tensor(full_completion_mask, dtype=torch.long),
        }

    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        chosen_items = []
        rejected_items = []

        for ex in examples:
            prompt_ids, chosen_ids, rejected_ids = ex["prompt_ids"], ex["chosen_ids"], ex["rejected_ids"]

            chosen_items.append(self._build_sequence(prompt_ids, chosen_ids))
            rejected_items.append(self._build_sequence(prompt_ids, rejected_ids))

        all_items = chosen_items + rejected_items

        input_ids_tensor = pad([x["input_ids"] for x in all_items], padding_value=self.pad_token_id, padding_side="right")
        attention_mask_tensor = pad([x["attention_mask"] for x in all_items], padding_value=0, padding_side="right")
        completion_mask_tensor = pad([x["completion_mask"] for x in all_items], padding_value=0, padding_side="right")


        batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "completion_mask": completion_mask_tensor,
        }

        if "ref_chosen_logps" in examples[0]:
            batch["ref_chosen_logps"] = torch.tensor([ex["ref_chosen_logps"] for ex in examples], dtype=torch.float)

        if "ref_rejected_logps" in examples[0]:
            batch["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples], dtype=torch.float)

        return batch




def dpo(args):
    datasets.logging.set_verbosity_error() # Disable hashing warning from calling .map in DPOTrainer

    # Load the data
    train_data, val_data = load_train_val_data(args["dataset"])

    # Load model
    base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
    model = get_model(base_model, False, pt_model_name=args["pt_model_name"], r=args["r"], lora_alpha=args["lora_alpha"], quantize=args["quantize"])
    if not args["pt_model_name"]:
        print("Using base model as reference model")

    # Train
    train_dpo_data = construct_dpo_data(train_data, args)
    val_dpo_data = construct_dpo_data(val_data, args)

    if args["wandb"]:
        wandb.login(key=args["wandb_key"], verify=True)
        wandb.init(project='dialogue_sim_dpo_mod')
        wandb.config.update(args, allow_val_change=True)
        print('Run id:', wandb.run.id)

    collator = StudentCompletionMaskCollator(tokenizer=tokenizer, pad_token_id=tokenizer.pad_token_id)

    config = DPOConfig(
        output_dir=get_checkpoint_path(args["model_name"]),
        num_train_epochs=args["epochs"],
        learning_rate=args["lr"],
        weight_decay=args["wd"],
        max_grad_norm=args["gc"],
        warmup_ratio=0.1,
        gradient_accumulation_steps=args["grad_accum_steps"],
        per_device_train_batch_size=args["train_batch_size"],
        per_device_eval_batch_size=args["val_batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        save_only_model=True,
        load_best_model_at_end=True,
        report_to="wandb" if args["wandb"] else "none",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=args["val_batch_size"],
        beta=args["beta"],
        max_length=MAX_LEN,
        truncation_mode=args["truncation_mode"]
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train_dpo_data,
        eval_dataset=val_dpo_data,
        processing_class=tokenizer,
        data_collator=collator
    )
    
    trainer.train()
    trainer.save_model()

    # Free up memory
    del trainer, base_model, model
    run_gc()

    # Test
    test({
        **args,
        **({"student_model": args["model_name"]} if args["role"] == "student" else {"tutor_model": args["model_name"]}),
        "temperature": None # Test-time temperature should be different from overgeneration temperature
    })

def main():
    initialize_seeds(221)

    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--dataset", default="eedi")
    parser.add_argument("--role", choices=["student", "tutor"], default="student")
    parser.add_argument("--subsample", type=float, default=0.2, help="Subsample dataset (0 for no subsampling), take from beginning of shuffle")
    parser.add_argument("--exclude_first_turns", type=int, default=5, help="Exclude first n turns of each dialogue")
    parser.add_argument("--test_on", choices=["val", "test"], default="val", help="Set to test on after training")
    parser.add_argument("--test_subsample", type=float, help="Subsample from test set")
    parser.add_argument("--random_form", action="store_true")
    # Settings
    parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
    parser.add_argument("--kcs_src", choices=["eedi"], default="eedi", help="Source of KCs for KT reward")
    parser.add_argument("--input_type", choices=["none", "profile", "dialogue"], default="none")
    parser.add_argument("--truncation_mode", choices=["keep_start", "keep_end"], default="keep_start")
    parser.add_argument("--wandb", action="store_true", help="Whether to log training with wandb")
    parser.add_argument("--wandb_key", type=str, default='ff70920d9852a9d2e78bbd1cd2e100154d2c9c7d', help="API key for Weights & Biases.")
    # Model
    parser.add_argument("--base_model", default="8b")
    parser.add_argument("--model_name")
    parser.add_argument("--pt_model_name")
    parser.add_argument("--quantize", action="store_true")
    # Training hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size at train-time")
    parser.add_argument("--val_batch_size", type=int, default=2, help="Batch size at validation-time")
    parser.add_argument("--grad_accum_steps", type=int, default=64, help="Steps to accumulate gradients for")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--gc", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--r", type=int, default=32, help="LoRA rank, only used if initializing from base model")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha, only used if initializing from base model")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta parameter")
    parser.add_argument("--epsilon", type=float, default=0.3, help="Score threshold for DPO pair forming")
    parser.add_argument("--num_pairs", type=int, default=4, help="Maximum number of pairs to form per turn")

    args = parser.parse_args().__dict__

    dpo(args)

if __name__ == "__main__":
    main()
