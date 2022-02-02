import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertModel, BertPreTrainedModel

from losses.focal_loss import FocalLoss
from .layers.crf import CRF


class BertCrfSoftmaxForJoint(BertPreTrainedModel):
    def __init__(self, config):
        super(BertCrfSoftmaxForJoint, self).__init__(config)
        ner_class_weight = [9.88289223e+00, 1.90443333e+02, 5.05154730e+01, 3.13229167e+00,
                            1.68117349e+00, 3.41296296e+01, 8.94100156e+00, 5.69235214e-01,
                            1.64967415e-01]
        ride_class_weight = [0.50111982, 223.75]
        visit_pos_weight = [2.389853137516689, 1.1745406824146982, 39.77777777777778]
        self.num_labels = config.ner_num_labels
        self.visit_num_labels = config.visit_num_labels
        self.ride_num_labels = config.ride_num_labels
        self.class_weight = torch.tensor(ner_class_weight, dtype=torch.float)
        self.ride_class_weight = torch.tensor(ride_class_weight, dtype=torch.float)
        self.visit_pos_weight = torch.tensor(visit_pos_weight, dtype=torch.float)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.ner_classifier = nn.Linear(config.hidden_size, config.ner_num_labels)
        self.visit_classifier = nn.Linear(config.hidden_size, config.visit_num_labels)
        self.ride_classifier = nn.Linear(config.hidden_size, config.ride_num_labels)
        self.crf = CRF(num_tags=config.ner_num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                labels=None, visit_labels=None, ride_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.ner_classifier(sequence_output)
        visit_logits = self.visit_classifier(sequence_output[:, 0, :])
        ride_logits = self.ride_classifier(sequence_output[:, 0, :])
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct_focal = FocalLoss(weight=self.class_weight.to(input_ids.device), ignore_index=0)
            loss_fct_ce = CrossEntropyLoss(weight=self.class_weight.to(input_ids.device), ignore_index=0)
            loss_visit_ce = BCEWithLogitsLoss(weight=self.visit_pos_weight.to(input_ids.device))
            loss_ride_focal = FocalLoss(weight=self.ride_class_weight.to(input_ids.device), ignore_index=0)
            loss_ride_ce = CrossEntropyLoss(weight=self.ride_class_weight.to(input_ids.device), ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                active_loss = attention_mask.contiguous().view(-1) == 1
                active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                active_labels = labels.contiguous().view(-1)[active_loss]
                loss_focal = loss_fct_focal(active_logits, active_labels)
                loss_ce = loss_fct_ce(active_logits, active_labels)
            else:
                loss_focal = loss_fct_focal(logits.contiguous().view(-1, self.num_labels), labels.view(-1))
                loss_ce = loss_fct_ce(logits.contiguous().view(-1, self.num_labels), labels.view(-1))

            # TODO: find focal weighted loss
            # loss_visit_focal = loss_visit_focal(visit_logits, visit_labels)
            loss_visit_ce = loss_visit_ce(visit_logits, visit_labels)
            loss_ride_focal = loss_ride_focal(ride_logits.view(-1, self.ride_num_labels), ride_labels.view(-1))
            loss_ride_ce = loss_ride_ce(ride_logits.view(-1, self.ride_num_labels), ride_labels.view(-1))
            loss_crf = self.crf(emissions=logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss_crf + 0.5 * (loss_focal + loss_ce +
                                              loss_visit_ce +
                                              loss_ride_focal + loss_ride_ce),) + outputs

        return outputs  # (loss), scores
