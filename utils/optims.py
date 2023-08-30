import torch.optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score


def make_optimizer(model, lr):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_group = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.05},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}    ]
    optimizer = torch.optim.AdamW(optimizer_group, lr=lr)
    criterion = nn.CrossEntropyLoss()
    return optimizer, criterion


def flat_accuracy(out, labels):
    out = out.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    pred = np.argmax(out, axis=1).flatten()
    labels_flatten = labels.flatten()
    length = labels_flatten.size
    ans = accuracy_score(pred, labels_flatten)
    num_correct = int(length * ans)
    return ans, num_correct
