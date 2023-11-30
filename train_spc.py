import torch
import numpy as np
import logging
import os
from tqdm import tqdm
from utils.dataset import read_data, make_dataset
from utils.model import load_model, save_model
from utils.optims import *
from transformers import logging as trans_log
from utils.Weighted_Loss import weighted_loss
from test import test_main
import argparse

trans_log.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_model', type=str, default='roberta')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--dataset', type=str, default='Laptops', choices=['Laptops', 'Restaurants', 'MAMS'])
    parser.add_argument('--normal_loss', type=str, default='CE')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_labels', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_len', type=int, default=96)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-5)
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def set_logger(args):
    LOG_FILE = 'Log/train_SPC.log'
    if not os.path.exists(LOG_FILE.split('/')[0]):
        os.mkdir(LOG_FILE.split('/')[0])
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


def train(
        train_set,
        val_set,
        optimizer,
        criterion,
        model,
        loss_function,
        device,
        num_epoch=20,
        loss_func_name='CE',
        dataset=''
):
    save_path = 'checkpoint/SPC/{}_best.ckpt'.format(dataset)
    path = save_path.rsplit('/', 1)[0]
    if not os.path.exists(path):
        os.mkdir(path)

    loss_func_lookup = {
        'CE': criterion,
        'BCE': loss_function.balanced_cross_entropy,
        'Focal': loss_function.focal_loss,
        'GHM': loss_function.ghm_loss
    }

    loss_func = loss_func_lookup[loss_func_name]
    best_acc = 0
    best_train_acc = 0

    logger.info('Start Training')
    for epoch in range(num_epoch):
        train_loss, val_loss = 0.0, 0.0
        train_acc, val_acc = 0.0, 0.0
        num_correct, covid_num_correct = 0, 0
        loop = tqdm(total=len(train_set), desc='Training Epoch {}'.format(epoch + 1), ncols=130, position=0, leave=True)
        model.train()
        for i, data in enumerate(train_set):
            loop.update()
            ids, mask, labels = [el.to(device) for el in data[:-1]]
            optimizer.zero_grad()
            label_out = model(ids, mask)

            loss = loss_func(label_out, labels)

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

            temp, _ = flat_accuracy(label_out, labels)
            train_acc += temp

            if i > 0:
                loop.set_postfix(loss=train_loss / i, train_acc=train_acc / i)

        loop.close()

        logger.info('Start Validation')
        model.eval()
        pred = []
        with torch.no_grad():
            for j, data in enumerate(val_set):
                val_ids, val_mask, val_labels = [el.to(device) for el in data[:-1]]
                val_label_out = model(val_ids, val_mask)
                loss = loss_func(val_label_out, val_labels)
                val_loss += loss.item()

                temp_acc, temp_correct = flat_accuracy(val_label_out, val_labels)
                pred += list(torch.max(val_label_out, dim=-1)[1].flatten().cpu().numpy())
                num_correct += temp_correct

        val_acc = num_correct / len(val_set.dataset.indices)
        logger.info('epoch:%d | train_loss:%f | train_acc:%f | val_loss:%f | val_acc:%f'
                    % (epoch + 1, train_loss / (i + 1), train_acc / (i + 1), val_loss / (j + 1), val_acc))

        if val_acc > best_acc:
            save_model(model, save_path, logger)
            best_acc = val_acc

        if train_acc < best_train_acc:
            break
        else:
            best_train_acc = train_acc
    return save_path


def main(args, logger):
    if args.base_model == 'roberta':
        eos = '</s>'

    logger.info('Dataset: {}'.format(args.dataset))
    train_sents, train_labels = read_data(args.dataset, 'train', eos)
    dev_sents, dev_labels = read_data(args.dataset, 'dev', eos)
    train_sents += dev_sents
    train_labels += dev_labels

    logger.info('Train Examples: {}'.format(len(train_sents)))

    test_sents, test_labels = read_data(args.dataset, 'test', eos)
    logger.info('Test Examples: {}'.format(len(test_sents)))
    test_labels = torch.LongTensor(test_labels)

    # Define Loss
    criterion = nn.CrossEntropyLoss()
    loss_function = weighted_loss(args.num_labels, args.batch_size, args.device, criterion=criterion)

    # Load model
    tokenizer, model = load_model(args.pretrained_model, args.num_labels)
    model.to(args.device)

    # Load Optimizer
    optimizer, criterion = make_optimizer(model, args.lr)

    # Make dataset
    train_set = make_dataset(tokenizer, train_sents, train_labels, args.batch_size, max_len=args.max_len)
    test_set = make_dataset(tokenizer, test_sents, test_labels, args.batch_size, max_len=args.max_len, shuffle=False)

    # Train
    save_path = train(train_set, test_set, optimizer, criterion, model, loss_function, args.device,
          num_epoch=args.epochs, loss_func_name=args.normal_loss, dataset=args.dataset)
    torch.cuda.empty_cache()

    # Test
    test_main(logger, args.dataset, save_path, args.device, args.pretrained_model)


if __name__ == "__main__":
    args = parse_args()
    logger = set_logger(args)
    logger.info('Training args: {}'.format(args))

    set_seed(args.seed)
    logger.info('Seed: {}'.format(args.seed))
    main(args, logger)
