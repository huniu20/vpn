import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from transformers import BertModel,BertConfig
from typing import Optional
from transformers.models.bert.modeling_bert import BertOnlyMLMHead,BertForPreTraining
from torchcrf import CRF
from torch.nn import CrossEntropyLoss


class Vpn(torch.nn.Module):
    def __init__(self, param) -> None:
        super().__init__()
        self.param = param
        # self.bert = BertModel.from_pretrained(param.dataset_cls.backbone)
        self.bert_for_pretraining = BertForPreTraining.from_pretrained(param.dataset_cls.backbone)
        self.crf_layer = CRF(param.num_tags, batch_first=True)
        self.entity_id_to_words_id = self.param.entity_id_to_word_id
        self.loss_fn = CrossEntropyLoss()
        print(self.entity_id_to_words_id)
        
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # batch_size * sequence_length * vocab_size
        prediction_logits = self.bert_for_pretraining(input_ids, attention_mask).prediction_logits
        
        # gather: vocab_size -> num_tags
        emission_scores = [None] * len(self.entity_id_to_words_id)
        # print(emission_scores)
        for entity_id, id_and_freq in self.entity_id_to_words_id.items():
            # print(id_and_freq)
            # batch_size * sequence_length
            words_id = torch.LongTensor(id_and_freq[0]).to(prediction_logits.device)
            words_freq = torch.FloatTensor(id_and_freq[1]).to(prediction_logits.device)
            if self.param.sum_op == "average":
                entity_logits = torch.sum(torch.index_select(prediction_logits, dim=-1, index=words_id), dim=-1)
            elif self.param.sum_op == "weighted":
                entity_logits = self.param.top_k * torch.sum(torch.index_select(prediction_logits, dim=-1,
                                                            index=words_id) * words_freq, dim=-1)
            else:
                # print(self.param.sum_op)
                raise ValueError("incorrect sum_op!")
            emission_scores[int(entity_id)] = entity_logits
        assert None not in emission_scores
        # print(emission_scores)
        output_logits = torch.stack(emission_scores, dim=-1)
        # output_logits_crf = self.crf_layer(output_logits)
        if labels is not None:
            if self.param.use_crf == True:
                crf_loss = - self.crf_layer(output_logits, labels, mask=attention_mask.byte())
                return crf_loss
            else:
                # print("shape:",output_logits[1,:,:],labels[1,:])
                # print("shape:",output_logits.reshape((-1,self.param.num_tags)).shape,labels.reshape((-1,)).shape)
                loss = self.loss_fn(output_logits.reshape((-1,self.param.num_tags)), labels.reshape((-1,)))
                return loss
        else:
            # print(type(torch.FloatTensor(output_logits)), "??",type(torch.FloatTensor(attention_mask)))
            if self.param.use_crf == True:
                output_label_seq_crf = self.crf_layer.decode(output_logits.float(), mask=attention_mask.byte())
                return output_label_seq_crf
            else:
                output_label = torch.argmax(output_logits,dim=-1)
                return output_label