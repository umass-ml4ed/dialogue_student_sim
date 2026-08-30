# Simulated Student in Dialogue

This code is associated with the paper <a href="https://arxiv.org/abs/2605.30051">Who Am I? History-Aware Profiles for Student Simulation in Tutoring Dialogues</a>. Currently, a slice of this data is used in an ongoing public-facing data mining challenge, which lasts until late August, we will release the data once the challenge is finished. 

If you find this code useful, please cite us!
```
@misc{duan2026ihistoryawareprofilesstudent,
      title={Who Am I? History-Aware Profiles for Student Simulation in Tutoring Dialogues}, 
      author={Zhangqi Duan and Shuyan Huang and Alexander Scarlatos and Jaewook Lee and Simon Woodhead and Andrew Lan},
      year={2026},
      eprint={2605.30051},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.30051}, 
}
```


## Recreating Annotations

Annotate the data using OpenAI:
```
python -m sim_student.annotate --label acts               # Dialogue acts (for Acts metric)
python -m sim_student.annotate --label corr               # Correctness (for Correctness, Errors, and Knowledge Acquistion metrics)
python -m sim_student.annotate --label eedi_kcs           # Turn-level KCs
```

## Training Models for Automated Metrics

Many of our automated metrics rely on fine-tuned models that will make predictions on simulated student turns. Run the following to train these models.

### Dialouge Act Classifier (Acts Metric)
```
python -m sim_student.acts train --model_name acts-8b
```

## Student Simulation

The following trains/tests/evaluates the student models implemented in this repo.

### Fine-tuning methods

Train SFT and test/evaluate on validation set:
```
python -m sim_student.sft --model_name eedi-stud-sft-8b --input_type profile --ks_num
```

Train DPO and test/evaluate on validation set:
```
python -m sim_student.turn_dpo --pt_model_name eedi-stud-sft-8b --model_name eedi-stud-dpo-8b --input_type profile --ks_num
```

Train SFT on Profile Agent for Student Simulation on validation set:
```
python -m sim_student.profile_agent --model_name profile_8b --ks_num
```

Train GRPO on Profile Agent on validation set:
```
python -m sim_student.profile_grpo --pt_model_name profile_8b --model_name profile_grpo_8b --dialogue_model_name eedi-stud-dpo-8b --input_type profile --ks_num
```

Test and evaluate on test set
```
python -m sim_student.testing --student_model eedi-stud-dpo-8b --input_type profile --ks_num
```
