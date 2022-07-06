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

## Citation

Please cite the following paper if you use or ideate with Multi-task Transformers in your work:

https://aclanthology.org/2021.eacl-main.171.pdf

> Ghosh, D., Shrivastava, R. and Muresan, S., 2021, April. “Laughing at you or with you”: The Role of Sarcasm in Shaping the Disagreement Space. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume (pp. 1998-2010).

```
@inproceedings{ghosh2021laughing,
  title={“Laughing at you or with you”: The Role of Sarcasm in Shaping the Disagreement Space},
  author={Ghosh, Debanjan and Shrivastava, Ritvik and Muresan, Smaranda},
  booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume},
  pages={1998--2010},
  year={2021}
}
```
