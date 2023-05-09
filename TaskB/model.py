import config
import torch
import transformers
import torch.nn as nn
from torchcrf import CRF

class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos, wandb_config = {'dropout' : 0}):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.dropout = wandb_config['dropout']
        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL_PATH,return_dict=False)
        self.bert_drop = nn.Dropout(self.dropout)
        self.out_fcl = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.out_tag = nn.Linear(768, self.num_tag)
        self.crf_tag = CRF(self.num_tag, batch_first=True)
        
    def forward(self, x, ids, mask, token_type_ids, target_pos, target_tag):
    
        output, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        
        output = self.bert_drop(output)
        output = self.out_fcl(output)
        output = self.tanh(output)
        output = self.bert_drop(output)
        emission_tag = self.out_tag(output)
        loss_tag = self.crf_tag(emission_tag, target_tag, mask=mask.bool(), reduction='mean')
        predicted_tag = self.crf_tag.decode(emission_tag, mask=mask.bool())
        
        loss = -1 * loss_tag
        
        return loss, predicted_tag
