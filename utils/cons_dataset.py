import numpy as np
import pandas as pd
import torch
import json
import re
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import xml.dom.minidom


def helper(term, sent):
    for i in range(len(sent) - len(term) + 1):
        if sent[i: i + len(term)] == term:
            return True
    return False


def find_term(text, pos, placeholder):
    ans = ' '.join([text[:pos[0]], placeholder, text[pos[1]:]])
    return ans


def segmentation_sents(text, terms, pos, placeholder):
    out_sents = []
    for i, term in enumerate(terms):
        sent = find_term(text, pos[i], placeholder)
        out_sents.append(text)
    return out_sents


def clean_list(sent_list):
    for i in range(len(sent_list)):
        if sent_list[i] == '\xa0':
            sent_list[i] = ' '
    return


def trans_pos(sent, pos_list):
    ans = []
    for pos in pos_list:
        temp_sent_list = [s for s in sent[:pos[0]]]
        clean_list(temp_sent_list)
        start = pos[0] - temp_sent_list.count(' ')
        temp_sent_list = [s for s in sent[:pos[1]]]
        clean_list(temp_sent_list)
        end = pos[1] - temp_sent_list.count(' ')
        ans.append([start, end])
    return ans


def get_term_mask(tokenizer, sent, term, pos_list):
    trans_pos_list = trans_pos(sent.replace('\xa0', ' '), pos_list)
    sent = sent.replace('  ', ' ')
    sent_tok_1 = tokenizer.tokenize(sent)
    sent_tok = [tokenizer.convert_tokens_to_string(s).strip() for s in sent_tok_1]
    assert len(sent_tok_1) == len(sent_tok)
    ans = np.zeros(len(sent_tok) + 2)
    single_term_mask = []
    trans_pos_list.sort(key=lambda x:x[0])
    for pos in trans_pos_list:
        mask = [0 for _ in range(len(sent_tok) + 2)]
        temp_pos = 0
        start, end = -1, -1
        for i, s in enumerate(sent_tok + ['']):
            if pos[0] == temp_pos:
                start = i
            if pos[1] <= temp_pos:
                end = i
            temp_pos += len(s)
            if start > -1 and end > -1:
                break
        if not -1 in [start, end]:
            mask[start + 1: end + 1] = [1] * (end - start)
            ans = ans + np.array(mask)
            single_term_mask.append(mask)
    ans.astype(np.int)
    return single_term_mask, list(ans), sent_tok_1


def read_data(dataset, tpe, tokenizer, placeholder='*', delimiter='</s>', spc=True):
    label_lookup = {
        'negative': 0,
        'positive': 2,
        'neutral': 1,
    }
    if dataset in ['Laptops', 'Restaurants']:
        return read_semeval_data(dataset, tpe, tokenizer, label_lookup,
                                 placeholder=placeholder, delimiter=delimiter, spc=spc)
    elif dataset == 'MAMS':
        return read_mams_data(dataset, tpe, tokenizer, label_lookup,
                                 placeholder=placeholder, delimiter=delimiter, spc=spc)


def read_mams_data(dataset, tpe, tokenizer, label_lookup, placeholder='*', delimiter='</s>', spc=True):
    path = 'data/{}/{}.xml'.format(dataset, tpe)
    Domtree = xml.dom.minidom.parse(path)
    collect = Domtree.documentElement
    sentences = collect.getElementsByTagName('sentence')
    labels_fix, sents_fix, terms_fix, mask_fix, pos_fix, all_term_mask_fix, is_multi = [], [], [], [], [], [], []
    for sent in sentences:
        text = sent.getElementsByTagName('text')[0].childNodes[0].data
        categories = sent.getElementsByTagName('aspectTerms')[0]
        terms, pos = [], []
        for cate in categories.childNodes:
            try:
                aspect = cate.getAttribute('term')
                label = cate.getAttribute('polarity')
                start = cate.getAttribute('from')
                end = cate.getAttribute('to')
                terms.append(aspect)
                pos.append([int(start), int(end)])
                labels_fix.append(label_lookup[label])
            except:
                continue

        single_term_mask, all_term_mask, sent_tok = get_term_mask(tokenizer, text, terms, pos)
        sents_fix += [sent_tok] * len(terms)
        terms_fix += terms
        mask_fix += single_term_mask
        all_term_mask_fix += [all_term_mask] * len(terms)
        pos_fix += pos
        is_multi += [len(terms) > 1] * len(terms)

        assert len(sent_tok) == len(single_term_mask[0]) - 2
        if len(sents_fix) != len(mask_fix):
            get_term_mask(tokenizer, text, terms, pos)

    return sents_fix, terms_fix, labels_fix, mask_fix, pos_fix, all_term_mask_fix, is_multi


def read_semeval_data(dataset, tpe, tokenizer, label_lookup, placeholder='*', delimiter='</s>', spc=True):
    data_path = 'data/SemEval 2014 Task 4/{}/{}_sent.json'.format(dataset, tpe)
    data = json.load(open(data_path, 'r', encoding='utf-8'))
    labels_fix, sents_fix, terms_fix, mask_fix, pos_fix, all_term_mask_fix = [], [], [], [], [], []
    is_multi = []
    for key in data.keys():
        item = data[key]
        sent = item['sentence']
        term_list = item['term_list']
        term = [term_list[key]['term'] for key in term_list.keys()]
        label = [label_lookup[term_list[key]['polarity']] for key in term_list.keys()]
        pos = [[term_list[key]['from'], term_list[key]['to']] for key in term_list.keys()]
        sents = segmentation_sents(sent, term, pos, placeholder)
        single_term_mask, all_term_mask, sent_tok = get_term_mask(tokenizer, sent, term, pos)
        try:
            labels_fix += label
            sents_fix += [sent_tok] * len(term)
            terms_fix += term
            mask_fix += single_term_mask
            all_term_mask_fix += [all_term_mask] * len(term)
            pos_fix += pos
            is_multi += [len(term) > 1] * len(term)
        except:
            pass

    if spc:
        spc_sent = [' '.join([sents_fix[i], delimiter,terms_fix[i]]) for i in range(len(sents_fix))]
        return spc_sent, labels_fix
    return sents_fix, terms_fix, labels_fix, mask_fix, pos_fix, all_term_mask_fix, is_multi


def read_dep_data(data_path):
    data = open(data_path, 'r', encoding='utf-8').readlines()
    sents, indexes, terms = [], [], []
    for da in data:
        sent, index, term = da.split('\t')
        sent = sent.split()
        index = [int(t) for t in index.split()]
        term = term.split()
        sents.append(sent)
        indexes.append(index)
        terms.append(term)
    return sents, indexes, terms


def read_unlabeled_data(data_path):
    data = pd.read_table(data_path, encoding='utf-8', names=['sent'])
    sents = list(data['sent'])
    return sents


def make_dataset(tokenizer, sents, labels, train_term_mask, batch_size, radio=1, max_len=64, shuffle=True):
    sents_tok = tokenizer(sents, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    datasets = DataToDataset(sents_tok, labels, train_term_mask, sents)
    train_size = int(len(datasets) * radio)
    test_size = len(datasets) - train_size
    train_set, val_set = random_split(dataset=datasets, lengths=[train_size, test_size])
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=1,
        shuffle=shuffle,
        collate_fn=train_set.dataset.collate    )
    return train_loader


def sequence_padding(mask_list, max_len):
    ans = [m + [0] * (max_len - len(m)) for m in mask_list]
    return ans


class DataToDataset(Dataset):
    def __init__(self, encoding, label, term_mask, sents):
        self.encoding = encoding
        self.label = label
        self.term_mask = term_mask
        self.sents= sents

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        return self.encoding['input_ids'][item], self.encoding['attention_mask'][item], \
               self.label[item], self.term_mask[item], self.sents[item]

    @staticmethod
    def collate(examples):
        batch_ids, batch_mask, batch_labels, batch_term_mask, batch_sents = [], [], [], [], []
        for example in examples:
            ids, mask, label, term, sent = example
            batch_ids.append(ids)
            batch_mask.append(mask)
            batch_labels.append(label)
            batch_term_mask.append(term)
            batch_sents.append(sent)

        batch_ids = torch.stack(batch_ids)
        batch_mask = torch.stack(batch_mask)
        batch_labels = torch.stack(batch_labels)
        batch_term_mask = sequence_padding(batch_term_mask, batch_ids.shape[1])
        batch_term_mask = torch.BoolTensor(batch_term_mask)
        return batch_ids, batch_mask, batch_labels, batch_term_mask, batch_sents