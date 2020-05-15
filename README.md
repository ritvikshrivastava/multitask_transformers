# Multi-task Transformers for Sequence Classification

Installation Steps: <br>

1. Install HuggingFace's [transformers](https://github.com/huggingface/transformers) library

2. Install multitask_transformers <br>
```
git clone https://github.com/ritvikshrivastava/multitask_transformers.git
cd multitask_transformers
pip install .
```

Run Roberta Multi-Task for Sarcasm Detection - Argument Relation Detection Task: <br>

```
cd multitask_transformers/scripts/
./run_sarc_arg.sh
```

Run Roberta Alternative Training Multi-Task for Sarcasm Detection - Argument Relation Detection Task: <br>

```
cd multitask_transformers/scripts/
./run_alternate_data_multitask.sh
```