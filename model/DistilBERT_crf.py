

from transformers import DistilBertModel, DistilBertPreTrainedModel
import torch.nn as nn
#from torch.nn.utils.rnn import pad_sequence
from .layers.crf import CRF
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
log_soft = F.log_softmax

class DistilBertCrfForNer(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertCrfForNer, self).__init__(config)
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs
        return outputs

class BertSoftmaxForNer(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilBertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
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

class BertSoftmaxForSpan(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForSpan, self).__init__(config)
        self.num_labels = config.num_labels
        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.start_fc = nn.Linear(config.hidden_size, 4)
        self.end_fc = nn.Linear(config.hidden_size, 4)
        reduction = 'none'
        self.criterion = FocalLoss(reduction=reduction)
        self.crition = LabelSmoothingCrossEntropy(reduction='none')
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, start_ids=None, end_ids=None):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        end_logits = self.end_fc(sequence_output)
        outputs = (start_logits, end_logits) + outputs[1:]  # add hidden states and attention if they are here
        if start_ids is not None and end_ids is not None:
            start_logits = start_logits.view(-1, 4)
            end_logits = end_logits.view(-1, 4)

            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_ids.view(-1)[active_loss]
            active_end_labels = end_ids.view(-1)[active_loss]
            start_loss = self.criterion(active_start_logits, active_start_labels).mean()
            end_loss = self.criterion(active_end_logits, active_end_labels).mean()
            loss = start_loss + end_loss
            outputs = (loss,) + outputs
        return outputs
