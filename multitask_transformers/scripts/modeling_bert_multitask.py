import logging
import math
import os

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable

from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_callable
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer


logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://cdn.huggingface.co/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://cdn.huggingface.co/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://cdn.huggingface.co/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://cdn.huggingface.co/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://cdn.huggingface.co/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://cdn.huggingface.co/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://cdn.huggingface.co/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://cdn.huggingface.co/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://cdn.huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://cdn.huggingface.co/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://cdn.huggingface.co/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "bert-base-german-dbmdz-cased": "https://cdn.huggingface.co/bert-base-german-dbmdz-cased-pytorch_model.bin",
    "bert-base-german-dbmdz-uncased": "https://cdn.huggingface.co/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    "bert-base-japanese": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese/pytorch_model.bin",
    "bert-base-japanese-whole-word-masking": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/pytorch_model.bin",
    "bert-base-japanese-char": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-char/pytorch_model.bin",
    "bert-base-japanese-char-whole-word-masking": "https://cdn.huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/pytorch_model.bin",
    "bert-base-finnish-cased-v1": "https://cdn.huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    "bert-base-finnish-uncased-v1": "https://cdn.huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
    "bert-base-dutch-cased": "https://cdn.huggingface.co/wietsedv/bert-base-dutch-cased/pytorch_model.bin",
}

def dynamic_loss(arg_loss,sarc_loss):
    sigma_1 = Variable(torch.tensor(0.5), requires_grad=True)
    sigma_2 = Variable(torch.tensor(0.5), requires_grad=True)
    arg_loss_dyn = torch.mul(torch.div(1.0, torch.mul(2.0, torch.square(sigma_1))), arg_loss) \
          + torch.log(torch.square(sigma_1))
           
    sarc_loss_dyn = torch.mul(torch.div(1.0, torch.mul(2.0, torch.square(sigma_2))), sarc_loss) \
          + torch.log(torch.square(sigma_2))
     
    loss = torch.add(arg_loss_dyn, sarc_loss_dyn)
   
    return loss

class BertForMultitaskSequenceClassification(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout_t1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_t1 = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout_t2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_t2 = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels_t1=None,
        labels_t2=None
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output_t1 = self.dropout_t1(pooled_output)
        logits_t1 = self.classifier_t1(pooled_output_t1)

        pooled_output_t2 = self.dropout_t2(pooled_output)
        logits_t2 = self.classifier_t2(pooled_output_t2)

        outputs = (logits_t1,) + (logits_t2,) + outputs[2:]  # (bs, seq_len, dim)

        if labels_t1 is not None and labels_t2 is not None:
            loss_fct = CrossEntropyLoss()
            loss_t1 = loss_fct(logits_t1.view(-1, self.num_labels), labels_t1.view(-1))
            loss_t2 = loss_fct(logits_t2.view(-1, self.num_labels), labels_t2.view(-1))

            #loss = loss_t1 + loss_t2
            loss = dynamic_loss(loss_t2, loss_t1)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class BertForAlternateMultitaskClassification(BertPreTrainedModel):
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.bert = BertModel(config)
        self.dropout_t1 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_t1 = nn.Linear(config.hidden_size, config.num_labels)
        self.dropout_t2 = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_t2 = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        task=None,
    ):
        
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        if task[0] == 0:
            pooled_output_t1 = self.dropout_t1(pooled_output)
            logits = self.classifier_t1(pooled_output_t1)

        elif task[0] == 1:
            pooled_output_t2 = self.dropout_t2(pooled_output)
            logits = self.classifier_t2(pooled_output_t2)

        outputs = (logits,) + (task[0], ) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
