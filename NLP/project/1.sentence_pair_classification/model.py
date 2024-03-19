import torch
import torch.nn as nn
from transformers import (
    BertPreTrainedModel,
    BertModel,
    AutoConfig,
    AdamW,
    get_scheduler,
)


class BertForPairwiseCLS(BertPreTrainedModel):
    def __init__(self, config):
        super().init(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 2)
        self.post_init()

    def forward(self, X, labels=None):
        outputs = self.bert(**X)
        cls_vectors = outputs.last_hidden_state[
            :, 0, :
        ]  # [batch_size, sequence_length, hidden_size]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return loss, logits
