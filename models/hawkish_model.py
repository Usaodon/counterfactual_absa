import torch.nn as nn
import torch
from transformers import RobertaModel, AutoModel
# from models.roberta_model import RobertaModel


class BertClassificationModel(nn.Module):
    def __init__(self, model_path, num_label):
        super(BertClassificationModel, self).__init__()
        if model_path.startswith('roberta'):
            self.bert = RobertaModel.from_pretrained(model_path)
        else:
            self.bert = AutoModel.from_pretrained(model_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.label_fc_link1 = nn.Linear(768, 1024)
        self.label_fc_link2 = nn.Linear(1024, 768)
        self.dropout = nn.Dropout(0.1)
        self.label_fc = nn.Linear(768, num_label)

    def forward(self, ids, mask=None, output_attentions=False):
        if output_attentions:
            attns = self.bert(input_ids=ids, attention_mask=mask, output_attentions=output_attentions).attentions
            return attns

        out = self.bert(input_ids=ids, output_attentions=output_attentions)
        if isinstance(self.bert, RobertaModel):
            out = out.pooler_output
        else:
            out = out['last_hidden_state'][:, 0, :]

        # 输出label
        out = self.label_fc_link1(out)
        out = self.label_fc_link2(out)
        label_out = self.label_fc(out)
        label_out = self.dropout(label_out)
        label_out = torch.softmax(label_out, dim=1)
        return label_out