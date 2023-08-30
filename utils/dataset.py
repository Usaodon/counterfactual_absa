import numpy as np
import pandas as pd
import torch
import json
import re
import xml.dom.minidom
from torch.utils.data import Dataset, random_split, DataLoader
from xml.dom.minidom import parse


def helper(term, sent):
    for i in range(len(sent) - len(term) + 1):
        if sent[i: i + len(term)] == term:
            return True
    return False


def find_term(sents, sents_pos, pos):
    for i in range(len(sents_pos) - 1):
        if sents_pos[i] <= pos[0] < pos[1] <= sents_pos[i + 1]:
            return sents[i]
    for i in range(len(sents_pos) - 1):
        if sents_pos[i] <= pos[0] <= sents_pos[i + 1]:
            start = i
    for i in range(len(sents_pos) - 1):
        if sents_pos[i] <= pos[1] <= sents_pos[i + 1]:
            end = i
    return ','.join(sents[start: end + 1])


def segmentation_sents(text, terms, pos):
    splitter = r'[,.;:!?]'
    sents = re.split(splitter, text)
    if sents[-1] == '':
        sents = sents[:-1]
    sents_pos = [len(','.join(sents[:i])) for i in range(len(sents) + 1)]
    out_sents = []
    for i, term in enumerate(terms):
        sent = find_term(sents, sents_pos, pos[i])
        out_sents.append(sent)
    return out_sents


def read_data(dataset, tpe, eos):
    label_lookup = {
        'negative': 0,
        'positive': 2,
        'neutral': 1,
    }
    if dataset in ['Restaurants', 'Laptops']:
        return read_semeval(dataset, tpe, label_lookup, eos)
    elif dataset == 'MAMS':
        return read_mams(dataset, tpe, label_lookup, eos)


def read_mams(dataset, tpe, label_lookup, eos):
    path = 'data/{}/{}.xml'.format(dataset, tpe)
    Domtree = xml.dom.minidom.parse(path)
    collect = Domtree.documentElement
    sentences = collect.getElementsByTagName('sentence')
    spc_sents, labels = [], []
    for sent in sentences:
        text = sent.getElementsByTagName('text')[0].childNodes[0].data
        categories = sent.getElementsByTagName('aspectTerms')[0]
        for cate in categories.childNodes:
            try:
                aspect = cate.getAttribute('term')
                label = cate.getAttribute('polarity')
                start = int(cate.getAttribute('from'))
                end = int(cate.getAttribute('to'))

                spc_sent = text + ' ' + eos + ' ' + aspect
                spc_sents.append(spc_sent)
                labels.append(label_lookup[label])
            except:
                continue
    return spc_sents, labels


def read_semeval(dataset, tpe, label_lookup, eos):
    data_path = 'data/SemEval 2014 Task 4/{}/{}_sent.json'.format(dataset, tpe)
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    labels_fix, sents_fix, terms_fix =[], [], []
    for key in data.keys():
        item = data[key]
        sent = item['sentence']
        term_list = item['term_list']
        term = [term_list[key]['term'] for key in term_list.keys()]
        label = [label_lookup[term_list[key]['polarity']] for key in term_list.keys()]
        pos = [[term_list[key]['from'], term_list[key]['to']] for key in term_list.keys()]
        sents = segmentation_sents(sent, term, pos)

        try:
            labels_fix += label
            sents_fix += [sent] * len(term)
            terms_fix += term
        except:
            pass
    spc_sent = [sents_fix[i] + ' ' + eos + ' ' + terms_fix[i] for i in range(len(sents_fix))]
    return spc_sent, labels_fix


def read_unlabeled_data(data_path):
    data = pd.read_table(data_path, encoding='utf-8', names=['sent'])
    sents = list(data['sent'])
    return sents


def read_DA_data(path):
    lines = open(path, 'r', encoding='utf-8').readlines()
    da_sents, da_labels = [], []
    for line in lines:
        sent, label, isflip = line.strip().split('\t')
        da_sents.append(sent)
        da_labels.append(int(label))
    return da_sents, da_labels


def make_dataset(tokenizer, sents, labels, batch_size, radio=1, max_len=64, shuffle=True):
    sents_tok = tokenizer(sents, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    datasets = DataToDataset(sents_tok, labels, sents)
    train_size = int(len(datasets) * radio)
    test_size = len(datasets) - train_size
    train_set, val_set = random_split(dataset=datasets, lengths=[train_size, test_size])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=1, shuffle=shuffle)
    return train_loader


class DataToDataset(Dataset):
    def __init__(self, encoding, label, sents):
        self.encoding = encoding
        self.label = label
        self.sents= sents

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.encoding['input_ids'][item], self.encoding['attention_mask'][item], self.label[item], self.sents[item]
