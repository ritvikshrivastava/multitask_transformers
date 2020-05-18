from transformers import AutoConfig, EvalPrediction
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from multitask_transformers.scripts.trainer import Trainer
from transformers.configuration_roberta import RobertaConfig
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional
from multitask_transformers.scripts.utils import InputFeaturesMultitask, RobertaCustomTokenizer, f1
from multitask_transformers.scripts.modeling_roberta_multitask import RobertaForMultitaskSequenceClassification as model_select
import numpy as np

import torch
import os 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STORED_DATA_PATH = '../data'

label_dict = {'sarc': 1, 'notsarc': 0, 'agree': 1, 'disagree': 1, 'neutral': 0}

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


class SarcArgDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

        # batch_encoding = self.tokenizer.batch_encode_plus(
        # [(example.split('\t')[0], example.split('\t')[1]) for example in self.data if example.split('\t')[0] and example.split('\t')[1]],
        # add_special_tokens=True, max_length=512, pad_to_max_length=True,
        # )

        batch_encoding = self.tokenizer.batch_encode_plus(
        [(example.split('\t')[0], example.split('\t')[1]) for example in self.data], add_special_tokens=True, max_length=512, pad_to_max_length=True,
        )

        self.features = []
        for i in range(len(self.data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            sarclab, arglab = self.data[i].split('\t')[2:4]
            if not sarclab or not arglab:
                continue
            feature = InputFeaturesMultitask(
                **inputs,
                labels_t1=label_dict[sarclab.strip('\t').strip('\n')],
                labels_t2=label_dict[arglab.strip('\t').strip('\n')])
            self.features.append(feature)

        for i, example in enumerate(self.data[:5]):
            logger.info("*** Example ***")
            logger.info("features: %s" % self.features[i])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def _use_cuda():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True


def _load_data(dtype='train.txt'):
    with open(os.path.join(STORED_DATA_PATH, dtype)) as f:
        data = f.readlines()
    return data


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    config = RobertaConfig.from_pretrained(
    'roberta-base',
    num_labels=2,
    )

    # Set seed
    set_seed(training_args.seed)

    tokenizer = RobertaCustomTokenizer.from_pretrained('roberta-base')
    model = model_select.from_pretrained('roberta-base', config=config)

    # Fetch Datasets
    train_set = SarcArgDataset(_load_data('train.v1.txt'), tokenizer) if training_args.do_train else None
    eval_dataset = SarcArgDataset(_load_data('dev_temp_v1.txt'), tokenizer) if training_args.do_eval else None

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return f1(preds, p.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        # if trainer.is_world_master():
        #     tokenizer.save_pretrained(training_args.output_dir)


    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_datasets = [eval_dataset]
        for eval_dataset in eval_datasets:
            result = trainer.evaluate(eval_dataset=eval_dataset)

            output_eval_file = os.path.join(
                training_args.output_dir, f"eval_results_.txt"
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    return results


def test_single_run():
    # testing a single pair
    input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", "dog is very cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
    labels_t1 = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    labels_t2 = torch.tensor([1]).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids, labels_t1=labels_t1, labels_t2=labels_t2)
    loss, logits1, logits2 = outputs[:3]

    print(outputs)


if __name__ == "__main__":
    main()
