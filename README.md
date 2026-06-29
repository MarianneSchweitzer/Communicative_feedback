# Analysis of contingency of feedback

## Data pre-processing

```
python prepare_conversation_pairs_with_subjects.py --data_path utterances.csv --subjects_path subjects.csv

```
This script produces 5 different caregiver-children conversation data files : one with children without brain lesions, one with children with (right and left hemisphere) lesions, one with left hemisphere lesion children, one with right hemisphere lesion children and one with all of the children. 

## Annotation

TBD with each data file if needed.  

Clarification request annotation :

```
python annotate_cf.py --data_path conversation_pairs_prepared_all.csv 

```

Semantic & syntactic alignment annotation

```
python annotate_syntactic_semantic_alignment.py --input_path conversation_pairs_prepared_all.csv 

```

Grammaticality annotation 

```
python TODO

```

## Results plotting 

Results plotting for clarification requests:

```
python TODO

```

Results plotting for semantic and syntactic alignment:

```
python create_alignment_contingency_results_plot.py --data_path annotated_conversations.csv 

```
