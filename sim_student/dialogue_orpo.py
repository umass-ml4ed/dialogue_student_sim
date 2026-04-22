import argparse
import ast
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch import nn
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments
from sim_student.data_loading import load_train_val_data
from sim_student.model import get_base_model, get_model
from sim_student.utils import get_checkpoint_path, initialize_seeds, run_gc
from sim_student.training_utils import MAX_LEN
from sim_student.prompting import STUDENT_SYS_PROMPT_SHORT_PROFILE
from pdb import set_trace
from sim_student.testing import test


END_OF_DIALOGUE = "<end_of_dialogue>"

BOH_TOKEN_ID = 128006
EOH_TOKEN_ID = 128007


def get_local_prompt(dialogue, role, ending_turn=None, negative=False, question=None, first_turn_utterance=None):
	assert role in ("student", "tutor")
	turns = [*dialogue["turns"]] # Copy to not modify original dialogue

    # Add special end of dialogue tag to end of last turn (except when doing full dialogue generation, i.e. done flag is present)
	if "done" not in dialogue:
		turns[-1] = {**turns[-1], "content": turns[-1]["content"] + END_OF_DIALOGUE}
	
	_ending_turn = ending_turn if ending_turn is not None else len(turns)

    # Create prompt context
	context = build_input_context(dialogue, question=question)

	if negative:
		context += first_turn_utterance
		return context

    # Include first turn if other role starts
	first_turn_utterance = ""
	alt_role_title = "Tutor" if role == "student" else "Student"
	if not turns or turns[0]["role"] == role:
		starting_turn = 0
		first_turn_utterance = f"(No First {alt_role_title} Turn)"
		context += first_turn_utterance
	else:
		starting_turn = 1
		first_turn_utterance = f"First {alt_role_title} Turn: {turns[0]['content']}"
		context += first_turn_utterance

    # Add remaining turns and construct prompt
	turns = turns[starting_turn : _ending_turn]
	output = [{"role": "assistant" if turn["role"] == role else "user", "content": turn["content"]} for turn in turns]

	return context, output, first_turn_utterance


def build_input_context(row, question=None) -> str:
	sections: List[str] = []

	profile = row["profile_prev"].replace("\n\n", " ").strip()
	sections.append(f"Student Profile:\n{profile}")
	
	prev_dialogue = ""
	for turn in row["turn_prev"]:
		role_title = "Student" if turn["role"] == "student" else "Tutor"
		prev_dialogue += f"{role_title}: {turn['content']}\n"

	sections.append(f"Previous Dialogue:\n{prev_dialogue}")
	question = question if question is not None else row['question']
	sections.append(f"Question:\n{question}")

	res = "\n\n".join(sections).strip() + "\n\n"
	return res


def construct_suri_orpo_pairs(df, tokenizer, pairs_per_row) -> List[Dict[str, str]]:
	pairs = []
	missing_cnt = 0

	df = pd.DataFrame(df)
	for idx in range(len(df)):
		row = df.iloc[idx]

		xw_context, output, first_turn_utterance = get_local_prompt(row, role="student")
		chosen_prompt_parts = [{"role": "system", "content": STUDENT_SYS_PROMPT_SHORT_PROFILE}, {"role": "user", "content": xw_context}]
		chosen_prompt = tokenizer.apply_chat_template(chosen_prompt_parts+output, tokenize=False, add_generation_prompt=False)

		if len(chosen_prompt) > MAX_LEN:
			missing_cnt += 1
			continue

		subset = df[df["studentID"] != row["studentID"]]
		# randomly select pairs_per_row negative examples from subset using pandas sampling
		negatives = subset.sample(n=pairs_per_row, random_state=42)

		for ind in range(len(negatives)):
			neg_row = negatives.iloc[ind]
			xl_context = get_local_prompt(neg_row, role="student", negative=True, question=row['question'], first_turn_utterance=first_turn_utterance)

			rejected_prompt_parts = [{"role": "system", "content": STUDENT_SYS_PROMPT_SHORT_PROFILE}, {"role": "user", "content": xl_context}]
			rejected_prompt = tokenizer.apply_chat_template(rejected_prompt_parts+output, tokenize=False, add_generation_prompt=False)

			if len(rejected_prompt) > MAX_LEN:
				missing_cnt += 1
				continue

			pairs.append(
				{
					"prompt_chosen": chosen_prompt,
					"prompt_rejected": rejected_prompt,
				}
			)
	print(f"Missing {missing_cnt} due to length.")
	return pairs

def tokenize_pair(item, tokenizer):
	res = {}

	for prompt_name in ("prompt_chosen", "prompt_rejected"):
		prompt = item[prompt_name]

		tokenized = tokenizer(prompt, add_special_tokens=False)
		input_ids, attention_mask = tokenized["input_ids"], tokenized["attention_mask"]
		labels = input_ids[:]

		ids = torch.tensor(input_ids, dtype=torch.long)
		boh_idxs = (ids == BOH_TOKEN_ID).nonzero()
		eoh_idxs = (ids == EOH_TOKEN_ID).nonzero()

		for i in range(eoh_idxs[2] + 1):
			labels[i] = -100

		for header_ct in range(3, len(boh_idxs), 2):
			# Mask labels between start of each user header to end of subsequent assistant header
			end_idx = eoh_idxs[header_ct + 1] if header_ct + 1 < len(eoh_idxs) else len(labels)-1
			for i in range(boh_idxs[header_ct], end_idx + 1):
				labels[i] = -100
			
		suffix = prompt_name.split("_")[1]
		res[f"{suffix}_input_ids"] = input_ids
		res[f"{suffix}_attention_mask"] = attention_mask
		res[f"{suffix}_labels"] = labels

	return res


def pad(seqs, pad_value):
	max_len = max(len(x) for x in seqs)
	out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
	for i, seq in enumerate(seqs):
		out[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
	return out


@dataclass
class SuriORPODataCollator:
	label_pad_token_id: int
	pad_token_id: int

	def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
		batch = {}
		for k in ["chosen_input_ids", "chosen_attention_mask", "chosen_labels", "rejected_input_ids", "rejected_attention_mask", "rejected_labels"]:
			pad_val = self.label_pad_token_id if "labels" in k else (0 if "attention_mask" in k else self.pad_token_id)
			batch[k] = pad([f[k] for f in features], pad_val)
		
		return batch


class SuriIORPOTrainer(Trainer):
	def __init__(self, *args, beta: float, label_pad_token_id: int, padding_value: int, **kwargs):
		super().__init__(*args, **kwargs)
		self.beta = beta
		self.label_pad_token_id = label_pad_token_id
		self.padding_value = padding_value

	@staticmethod
	def get_batch_logps(
		logits: torch.FloatTensor,
		labels: torch.LongTensor,
		average_log_prob: bool,
		label_pad_token_id: int,
	) -> torch.FloatTensor:

		labels = labels[:, 1:].clone()
		logits = logits[:, :-1, :]
		loss_mask = labels != label_pad_token_id
		labels[labels == label_pad_token_id] = 0
		per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
		if average_log_prob:
			denom = loss_mask.sum(-1).clamp(min=1)
			return (per_token_logps * loss_mask).sum(-1) / denom
		return (per_token_logps * loss_mask).sum(-1)

	@staticmethod
	def _pad_to_length(x: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
		if x.shape[1] >= length:
			return x
		pad = torch.full((x.shape[0], length - x.shape[1]), pad_value, dtype=x.dtype, device=x.device)
		return torch.cat([x, pad], dim=1)

	def concatenated_inputs(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
		max_len = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])
		out = {}
		for prefix in ["chosen", "rejected"]:
			for field, pad_value in [("input_ids", self.padding_value), ("attention_mask", 0), ("labels", self.label_pad_token_id)]:
				key = f"{prefix}_{field}"
				padded = self._pad_to_length(batch[key], max_len, pad_value)
				out_key = f"concatenated_{field}"
				if out_key not in out:
					out[out_key] = padded
				else:
					out[out_key] = torch.cat([out[out_key], padded], dim=0)
		
		return out

	def odds_ratio_loss(self, chosen_logps: torch.FloatTensor, rejected_logps: torch.FloatTensor) -> torch.FloatTensor:
		eps = 1e-8

		chosen_p = torch.exp(chosen_logps).clamp(max=1 - eps)
		rejected_p = torch.exp(rejected_logps).clamp(max=1 - eps)

		log_odds = (chosen_logps - rejected_logps) - (torch.log1p(-chosen_p) - torch.log1p(-rejected_p))
		return self.beta * torch.log(torch.sigmoid(log_odds) + eps)


	def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False, num_items_in_batch: Optional[int] = None):
		concat = self.concatenated_inputs(inputs)
		outputs = model(input_ids=concat["concatenated_input_ids"], attention_mask=concat["concatenated_attention_mask"], use_cache=False)
		logits = outputs.logits

		bs = inputs["chosen_input_ids"].shape[0]
		chosen_logits = logits[:bs]
		rejected_logits = logits[bs:]
		chosen_labels = concat["concatenated_labels"][:bs]
		rejected_labels = concat["concatenated_labels"][bs:]

		ce_logits = chosen_logits[:, :-1, :].contiguous().view(-1, chosen_logits.shape[-1])
		ce_labels = chosen_labels[:, 1:].contiguous().view(-1)

		nll_loss = F.cross_entropy(ce_logits, ce_labels, ignore_index=self.label_pad_token_id)

		chosen_logps = self.get_batch_logps(chosen_logits, chosen_labels, True, self.label_pad_token_id)
		rejected_logps = self.get_batch_logps(rejected_logits, rejected_labels, True, self.label_pad_token_id)

		orpo_term = self.odds_ratio_loss(chosen_logps, rejected_logps).mean()
		loss = nll_loss - orpo_term

		if return_outputs:
			return loss, outputs
		return loss


def build_hf_dataset(pairs, tokenizer):
	tokenized = [tokenize_pair(item=item, tokenizer=tokenizer) for item in pairs]
	return HFDataset.from_list(tokenized)


def train_dialogue_suri_orpo(args: dict):
	train_data, val_data = load_train_val_data(args["dataset"])

	base_model, tokenizer = get_base_model(args["base_model"], args["quantize"])
	model = get_model(
		base_model,
		test=False,
		pt_model_name=args["pt_model_name"],
		r=args["r"],
		lora_alpha=args["lora_alpha"],
		quantize=args["quantize"],
	)

	train_pairs = construct_suri_orpo_pairs(df=train_data, tokenizer=tokenizer, pairs_per_row=args["pairs_per_row"])
	val_pairs = construct_suri_orpo_pairs(df=val_data, tokenizer=tokenizer, pairs_per_row=args["pairs_per_row"])

	train_dataset = build_hf_dataset(train_pairs, tokenizer)
	eval_dataset = build_hf_dataset(val_pairs, tokenizer)

	label_pad_token_id = -100
	collator = SuriORPODataCollator(label_pad_token_id=label_pad_token_id, pad_token_id=tokenizer.pad_token_id)

	output_dir = get_checkpoint_path(args["model_name"])
	training_args = TrainingArguments(
		output_dir=output_dir,
		num_train_epochs=args["epochs"],
		learning_rate=args["lr"],
		per_device_train_batch_size=args["train_batch_size"],
		per_device_eval_batch_size=args["val_batch_size"],
		gradient_accumulation_steps=args["grad_accum_steps"],
		weight_decay=args["wd"],
		max_grad_norm=args["gc"],
		warmup_ratio=0.1,
		# bf16=torch.cuda.is_available(),
		logging_steps=args["logging_steps"],
		save_strategy="epoch",
		eval_strategy="epoch" if eval_dataset is not None else "no",
		load_best_model_at_end=eval_dataset is not None,
		remove_unused_columns=False,
		label_names=["chosen_labels", "rejected_labels"],
		prediction_loss_only=True,
		report_to="none",
		save_total_limit=1,
	)

	trainer = SuriIORPOTrainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=collator,
		beta=args["beta"],
		label_pad_token_id=label_pad_token_id,
		padding_value=tokenizer.pad_token_id,
	)

	print(f"Constructed {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs.")
	trainer.train()
	trainer.save_model()

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
	parser.add_argument("--dataset", default="eedi")
	parser.add_argument("--role", choices=["student", "tutor"], default="student")
	parser.add_argument("--persona", choices=["none", "ocean", "freeform"], default="none")
	parser.add_argument("--model_name")
	parser.add_argument("--base_model", default="8b")
	parser.add_argument("--pt_model_name")
	parser.add_argument("--quantize", action="store_true")
	parser.add_argument("--r", type=int, default=32)
	parser.add_argument("--lora_alpha", type=int, default=64)
	parser.add_argument("--input_type", choices=["none", "profile", "dialogue"], default="none")
	parser.add_argument("--pairs_per_row", type=int, default=2)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--lr", type=float, default=5e-6)
	parser.add_argument("--wd", type=float, default=1e-2)
	parser.add_argument("--gc", type=float, default=1.0)
	parser.add_argument("--beta", type=float, default=0.1)

	parser.add_argument("--train_batch_size", type=int, default=1)
	parser.add_argument("--val_batch_size", type=int, default=2)
	parser.add_argument("--grad_accum_steps", type=int, default=64)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--truncate", type=int)

	args = parser.parse_args().__dict__

	train_dialogue_suri_orpo(args)


if __name__ == "__main__":
	main()
