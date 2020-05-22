import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import DistilBertConfig
from transformers.modeling_bert import BertEmbeddings, BertLayerNorm, BertModel, BertPreTrainedModel, gelu
from transformers.modeling_distilbert import DistilBertModel
from copy import deepcopy

logger = logging.getLogger(__name__)

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "distilbert-base-uncased": "https://cdn.huggingface.co/distilbert-base-uncased-pytorch_model.bin",
    "distilbert-base-uncased-distilled-squad": "https://cdn.huggingface.co/distilbert-base-uncased-distilled-squad-pytorch_model.bin",
    "distilbert-base-cased": "https://cdn.huggingface.co/distilbert-base-cased-pytorch_model.bin",
    "distilbert-base-cased-distilled-squad": "https://cdn.huggingface.co/distilbert-base-cased-distilled-squad-pytorch_model.bin",
    "distilbert-base-german-cased": "https://cdn.huggingface.co/distilbert-base-german-cased-pytorch_model.bin",
    "distilbert-base-multilingual-cased": "https://cdn.huggingface.co/distilbert-base-multilingual-cased-pytorch_model.bin",
    "distilbert-base-uncased-finetuned-sst-2-english": "https://cdn.huggingface.co/distilbert-base-uncased-finetuned-sst-2-english-pytorch_model.bin",
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


class DistilBertForMultitaskSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.pre_classifier_t1 = nn.Linear(config.dim, config.dim)
        self.pre_classifier_t2 = nn.Linear(config.dim, config.dim)
        self.classifier_t1 = nn.Linear(config.dim, config.num_labels)
        self.classifier_t2 = nn.Linear(config.dim, config.num_labels)
        self.dropout_t1 = nn.Dropout(config.seq_classif_dropout)
        self.dropout_t2 = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels_t1=None,
        labels_t2=None
        ):
        
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)

        pooled_output_t1 = hidden_state[:, 0]  # (bs, dim)
        pooled_output_t1 = self.pre_classifier_t1(pooled_output_t1)  # (bs, dim)
        pooled_output_t1 = nn.ReLU()(pooled_output_t1)  # (bs, dim)
        pooled_output_t1 = self.dropout_t1(pooled_output_t1)  # (bs, dim)
        logits_t1 = self.classifier_t1(pooled_output_t1)

        pooled_output_t2 = hidden_state[:, 0]  # (bs, dim)
        pooled_output_t2 = self.pre_classifier_t2(pooled_output_t2)  # (bs, dim)
        pooled_output_t2 = nn.ReLU()(pooled_output_t2)  # (bs, dim)
        pooled_output_t2 = self.dropout_t2(pooled_output_t2)  # (bs, dim)
        logits_t2 = self.classifier_t2(pooled_output_t2)

        outputs = (logits_t1,) + (logits_t2,) + distilbert_output[1:]
        if labels_t1 is not None and labels_t2 is not None:
            loss_fct = CrossEntropyLoss()
            loss_t1 = loss_fct(logits_t1.view(-1, self.num_labels), labels_t1.view(-1))
            loss_t2 = loss_fct(logits_t2.view(-1, self.num_labels), labels_t2.view(-1))

            #loss = loss_t1 + loss_t2
            loss = dynamic_loss(loss_t2, loss_t1)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DistilBertForAlternateMultitaskClassification(BertPreTrainedModel):
    config_class = DistilBertConfig
    pretrained_model_archive_map = DISTILBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "distilbert"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.distilbert = DistilBertModel(config)

        self.pre_classifier_t1 = nn.Linear(config.dim, config.dim)
        self.pre_classifier_t2 = nn.Linear(config.dim, config.dim)
        self.classifier_t1 = nn.Linear(config.dim, config.num_labels)
        self.classifier_t2 = nn.Linear(config.dim, config.num_labels)
        self.dropout_t1 = nn.Dropout(config.seq_classif_dropout)
        self.dropout_t2 = nn.Dropout(config.seq_classif_dropout)

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
        
        distilbert_output = self.distilbert(
            input_ids=input_ids, attention_mask=attention_mask, head_mask=head_mask, inputs_embeds=inputs_embeds
        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)

        if task[0] == 0:
            pooled_output_t1 = hidden_state[:, 0]  # (bs, dim)
            pooled_output_t1 = self.pre_classifier_t1(pooled_output_t1)  # (bs, dim)
            pooled_output_t1 = nn.ReLU()(pooled_output_t1)  # (bs, dim)
            pooled_output_t1 = self.dropout_t1(pooled_output_t1)  # (bs, dim)
            logits = self.classifier_t1(pooled_output_t1)

        elif task[0] == 1:
            pooled_output_t2 = hidden_state[:, 0]  # (bs, dim)
            pooled_output_t2 = self.pre_classifier_t2(pooled_output_t2)  # (bs, dim)
            pooled_output_t2 = nn.ReLU()(pooled_output_t2)  # (bs, dim)
            pooled_output_t2 = self.dropout_t2(pooled_output_t2)  # (bs, dim)
            logits = self.classifier_t2(pooled_output_t2)

        outputs = (logits,) + (task[0], ) + distilbert_output[1:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
