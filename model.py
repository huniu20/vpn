import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from transformers import BertModel,BertConfig
from typing import Optional
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class Vpn(torch.nn.Module):
    def __init__(self, param) -> None:
        super().__init__()
        config = BertConfig.from_pretrained(param.dataset_cls.backbone)
        self.bert = BertModel.from_pretrained(param.dataset_cls.backbone)
        
        
    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        pass