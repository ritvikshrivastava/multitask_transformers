# Multi-task Transformers for Sequence Classification

Installation Steps: <br>

1. Install HuggingFace's [transformers](https://github.com/huggingface/transformers) library

2. Install multitask_transformers <br>
```
git clone https://github.com/ritvikshrivastava/multitask_transformers.git
cd multitask_transformers
pip install .
```

Run DistilBert/Bert/RoBERTa Multi-Task for Sarcasm Detection - Argument Relation Detection Task: <br>

```
cd multitask_transformers/scripts/
./run_sarc_arg.sh
```
update model type in ``run_sarc_arg.sh`` by updating: <br> ``--model_name_or_path <pretrained/finetuned model name or path to dir>`` <br>

Run DistilBert/Bert/RoBERTa Alternative Training Multi-Task for Sarcasm Detection - Argument Relation Detection Task: <br>

```
cd multitask_transformers/scripts/
./run_alternate_data_multitask.sh
```
update model type in ``run_alternate_data_multitask.sh`` by updating <br> ``--model_name_or_path <pretrained/finetuned model name or path to dir>`` <br>