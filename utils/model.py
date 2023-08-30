import torch
import os
from transformers import RobertaTokenizer, AutoTokenizer
from models.hawkish_model import BertClassificationModel


def load_model(model_path, num_label):
    if model_path.startswith('roberta'):
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertClassificationModel(model_path, num_label)
    return tokenizer, model


def save_model(model, save_path, logger):
    torch.save(model, save_path)
    logger.info('Save checkpoint to '+ save_path)