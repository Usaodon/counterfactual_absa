import torch
import logging
import argparse
from utils.dataset import read_data
from sklearn.metrics import classification_report
from utils.optims import *
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_model', type=str, default='roberta')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--save_path', type=str, default='checkpoint/SPC/Laptops_best.ckpt')
    parser.add_argument('--dataset', type=str, default='Laptops', choices=['Laptops', 'Restaurants', 'MAMS'])
    args = parser.parse_args()
    return args


def set_logger(args):
    LOG_FILE = 'Log/test_{}.log'.format(args.dataset)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setLevel('INFO')
    console_handler.setLevel('INFO')

    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger('updateSecurity')
    logger.setLevel('INFO')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(sents, labels, model, tokenizer, device, logger):
    pred = []
    model.eval()
    with torch.no_grad():
        for sent in sents:
            tok = tokenizer(sent, padding='max_length', truncation=True, max_length=96, return_tensors='pt')
            ids, mask = tok['input_ids'].to(device), tok['attention_mask'].to(device)
            logit = torch.max(model(ids, mask), dim=-1)[1].item()
            pred.append(logit)

    logger.info('Performace\n' + classification_report(labels, pred, digits=4))


def test_main(logger, dataset, save_path, device, pretrained_model):
    base_model = 'roberta'
    if base_model == 'roberta':
        eos = '</s>'

    # Read test data
    sents, labels = read_data(dataset, 'test', eos)

    # Load model
    model = torch.load(save_path, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Test
    test(sents, labels, model, tokenizer, device, logger)


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger(args)
    logger.info('Testing args: {}'.format(args))
    test_main(logger, args.dataset, args.save_path, args.device, args.pretrained_model)
=======
import torch
import logging
import argparse
from utils.dataset import read_data
from sklearn.metrics import classification_report
from utils.optims import *
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_model', type=str, default='roberta')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--save_path', type=str, default='checkpoint/SPC/Laptops_best.ckpt')
    parser.add_argument('--dataset', type=str, default='Laptops', choices=['Laptops', 'Restaurants', 'MAMS'])
    args = parser.parse_args()
    return args


def set_logger(args):
    LOG_FILE = 'Log/test_{}.log'.format(args.dataset)
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    console_handler = logging.StreamHandler()
    file_handler.setLevel('INFO')
    console_handler.setLevel('INFO')

    formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger = logging.getLogger('updateSecurity')
    logger.setLevel('INFO')
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test(sents, labels, model, tokenizer, device, logger):
    pred = []
    model.eval()
    with torch.no_grad():
        for sent in sents:
            tok = tokenizer(sent, padding='max_length', truncation=True, max_length=96, return_tensors='pt')
            ids, mask = tok['input_ids'].to(device), tok['attention_mask'].to(device)
            logit = torch.max(model(ids, mask), dim=-1)[1].item()
            pred.append(logit)

    logger.info('Performace\n' + classification_report(labels, pred, digits=4))


def test_main(logger, dataset, save_path, device, pretrained_model):
    base_model = 'roberta'
    if base_model == 'roberta':
        eos = '</s>'

    # Read test data
    sents, labels = read_data(dataset, 'test', eos)

    # Load model
    model = torch.load(save_path, map_location=device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Test
    test(sents, labels, model, tokenizer, device, logger)


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger(args)
    logger.info('Testing args: {}'.format(args))
    test_main(logger, args.dataset, args.save_path, args.device, args.pretrained_model)
