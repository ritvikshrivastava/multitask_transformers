import dataclasses
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union
from transformers import RobertaTokenizer
from transformers import EvalPrediction
from transformers.tokenization_roberta import VOCAB_FILES_NAMES
from sklearn.metrics import f1_score, classification_report


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }

def f1(preds,labels):
    confusion_matrix = classification_report(labels, preds)
    acc = simple_accuracy(preds, labels)
    return{
        "f1": confusion_matrix,
        "acc": acc
    }

# def store_preds(p: EvalPrediction) -> Dict:
#     preds = np.argmax(p.predictions, axis=1)
#     return preds, p.label_ids

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    train_file: str
    eval_file: str
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    # overwrite_cache: bool = field(
    #     default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    # )


@dataclass(frozen=True)
class InputFeaturesMultitask:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    labels_t1: Optional[Union[int, float]] = None
    labels_t2: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


@dataclass(frozen=True)
class InputFeaturesAlternate:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: (Optional) Segment token indices to indicate first and second
            portions of the inputs. Only some models use them.
        label: (Optional) Label corresponding to the input. Int for classification problems,
            float for regression problems.
    """

    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    labels: Optional[Union[int, float]] = None
    task: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class RobertaCustomTokenizer(RobertaTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        merges_file,
        **kwargs):

        super().__init__(vocab_file, merges_file, **kwargs)

    def prepare_for_tokenization(self, text, add_special_tokens=False, **kwargs):
        if "add_prefix_space" in kwargs:
            add_prefix_space = kwargs["add_prefix_space"]
        else:
            add_prefix_space = add_special_tokens

        if add_prefix_space and text and not text[0].isspace():
            text = " " + text
        return text
