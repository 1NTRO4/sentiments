from transformers import RobertaModel, RobertaConfig, RobertaPreTrainedModel
import torch.nn as nn
from .layers.crf import CRF
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

# Roberta-CRF model
class RobertaCrfForNer(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaCrfForNer, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs

class RobertaSoftmaxForNer(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[1:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs
