# Simulated Student in Dialogue

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
