This work is expanding the work of Nikolaus & Fourtassi (2026) and is partly based on the following repo : https://github.com/mitjanikolaus/lm_feedback

# Analysis of contingency of feedback

## Data pre-processing

```
python prepare_conversation_pairs_with_subjects.py --data_path utterances.csv --subjects_path subjects.csv

```
This step produces five caregiver–child conversation datasets: children without brain lesions, children with brain lesions (all), children with left hemisphere lesions, children with right hemisphere lesions, all children combined.

## Annotation

Prepare conversation datasets and generate grouped corpora. 

Clarification request annotation:

```
python annotate_cf.py --data_path conversation_pairs_prepared_all.csv 

```

Semantic & syntactic alignment annotation:

```
python annotate_syntactic_semantic_alignment.py --input_path conversation_pairs_prepared_all.csv 

```

Grammaticality annotation:

```
python annotate_grammaticality.py --eval_model_path models/grammar_eval/version_19 --data_path conversation_pairs_prepared_all.csv

```

## Results plotting 

Results plotting for clarification requests:

```
python create_cr_contingency_results_plot.py --data_path annotated_conversations.csv 

```

Results plotting for semantic and syntactic alignment:

```
python create_alignment_contingency_results_plot.py --data_path annotated_conversations.csv 

```

# Analysis of effect of caregiver feedback on learning (typical & atypical children)

## Caregiver utterances preparation


```
python create_lm_corpus.py --input conversation_pairs_all_groups_annotated.csv --output caregiver_utterances_all_groups.csv

```
## Train LM Baseline

```
python train_lm.py fit --trainer.devices [0] --trainer.accelerator gpu --trainer.logger=WandbLogger --trainer.logger.name baseline_lesion_corpus --data.lm_data_path caregiver_utterances_all_groups.csv

```
