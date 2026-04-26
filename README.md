# Simulated Student in Dialogue

## Recreating Annotations

Annotate the data using OpenAI:
```
python -m sim_student.annotate --label questions          # Question solutions (for all metrics and student models)
python -m sim_student.annotate --label acts               # Dialogue acts (for Acts metric)
python -m sim_student.annotate --label corr               # Correctness (for Correctness, Errors, and Knowledge Acquistion metrics)
python -m sim_student.annotate --label eedi_kcs           # Turn-level KCs (for Knowledge Acquistion metric)
python -m sim_student.annotate --label ocean_personas     # OCEAN personas (for Knowledge Acquistion metric and OCEAN prompting method)
python -m sim_student.annotate --label freeform_personas  # Oracle summaries/personas (for Oracle and ICL prompting methods)
```

## Training Models for Automated Metrics

Many of our automated metrics rely on fine-tuned models that will make predictions on simulated student turns. Run the following to train these models.

### Dialouge Act Classifier (Acts Metric)
```
python -m sim_student.acts train --model_name acts-8b
```

### Correctness Classifier (Optional, only used for distribution analysis)
```
python -m sim_student.correctness train --model_name correctness-8b
```

## Student Simulation

The following trains/tests/evaluates the student models implemented in this repo.

### Fine-tuning methods

Train SFT and test/evaluate on validation set:
```
python -m sim_student.sft --model_name eedi-stud-sft-8b --input_type profile
```

Train DPO and test/evaluate on validation set:
```
python -m sim_student.dpo --pt_model_name eedi-stud-sft-8b --model_name eedi-stud-dpo-8b --input_type profile
```

Train SFT on Profile Agent on validation set:
```
python -m sim_student.profile_agent --model_name profile_8b
```

Train GRPO on Profile Agent on validation set:
```
python -m sim_student.profile_grpo --pt_model_name profile_8b --model_name profile_grpo_8b --dialogue_model_name eedi-stud-dpo-8b --input_type profile
```


### Prompting methods

Test and evaluate on test set:
```
python -m sim_student.testing --test_on test --baseline zs-eth                                 # Zero-Shot
python -m sim_student.testing --test_on test --baseline persona-ocean                          # OCEAN persona
python -m sim_student.testing --test_on test --baseline icl                                    # ICL
python -m sim_student.testing --test_on test --baseline reasoning --baseline_model gpt-5-mini  # Reasoning
python -m sim_student.testing --test_on test --baseline persona-ff                             # Oracle
```
