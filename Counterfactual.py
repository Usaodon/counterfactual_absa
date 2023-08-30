import argparse
import copy
import pandas as pd
import logging
import warnings
import os
from utils.optims import *
from transformers import RobertaTokenizer, AutoTokenizer
from nltk.corpus import wordnet
from captum.attr import LayerIntegratedGradients
from utils.pet import add_patten
from gen_aug_T5 import *
from tqdm import tqdm


warnings.filterwarnings('ignore')
label_lookup = {
    0: 'negative',
    2: 'positive',
    1: 'neutral'
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--base_model', type=str, default='roberta')
    parser.add_argument('--pretrained_model', type=str, default='roberta-base')
    parser.add_argument('--model_path', type=str, default='checkpoint/SPC/Laptops_best.ckpt')
    parser.add_argument('--dataset', type=str, default='Laptops', choices=['Laptops', 'Restaurants', 'MAMS'])
    parser.add_argument('--output_dir', type=str, default='data_aug')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_return_sequences', type=int, default=5)
    parser.add_argument('--therehold', type=float, default=0.7)
    args = parser.parse_args()
    return args


def set_logger(args):
    LOG_FILE = 'Log/Counterfactual-{}.log'.format(args.dataset)
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


def get_antoyms(word, pad):
    antonyms = [pad]
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return antonyms


def save_to_excel(path, sents, labels, normal_pred):

    querys = [sents[i].split('[SEP]')[-1] for i in range(len(sents))]
    sents = [sents[i].split('[SEP]')[0] for i in range(len(sents))]
    normal_pred = [label_lookup[n] for n in normal_pred]
    labels = [label_lookup[n] for n in labels]
    data = pd.DataFrame({
        '句子': sents,
        '方面词': querys,
        '预测': normal_pred,
        '标签': labels
    })
    data.to_excel(path)


def reconstruct_sent(sent, tokenizer, antoyms, eye, index):
    placeholder = '<unk>'
    sent[index + 1] = placeholder
    sent_encode = tokenizer.encode(sent[1:-1])[1:-1]
    sent_decode = tokenizer.decode(sent_encode)
    if index == 0:
        antoyms = antoyms.strip()
    sent_decode = sent_decode.replace(placeholder, antoyms)
    sent_tok = tokenizer.tokenize(sent_decode)
    add = len(sent_tok) - len(sent) + 2
    if add > 0:
        new_eye = torch.zeros(eye.shape[0] + add, eye.shape[1] + add).to(eye.device)
        new_eye[:index + 1, :index + 1] = eye[:index + 1, :index + 1]
        new_eye[index + 1 + add:, index + 1 + add:] = eye[index + 1:, index + 1:]
        return sent_decode, new_eye
    eye[index][index + 1] = 0
    return sent_decode, eye


def summerize_attributions(attr):
    attr = attr.sum(dim=-1).squeeze(0)
    attr = attr / torch.norm(attr)
    return attr


def inference(sent, base_sent, tokenizer, device, cur_label, ig):
    sent_pt = [tokenizer.vocab[s] for s in sent]
    base_sent = [tokenizer.vocab[s] for s in base_sent]
    input = torch.LongTensor(sent_pt).to(device).unsqueeze(0)
    baseline = torch.LongTensor(base_sent).to(device).unsqueeze(0)

    attributions = ig.attribute(inputs=input, baselines=baseline, target=cur_label, return_convergence_delta=False)
    attr = summerize_attributions(attributions)
    return attr


def main(logger, args):
    # Load T5 model
    t5_model_path = 't5-base'
    t5_tokenizer, t5_model = init_model(t5_model_path)
    t5_model.to(args.device)
    t5 = T5Aug(t5_model_path, t5_tokenizer, t5_model, args.device)

    # Read training set
    train_sents, train_terms, train_labels, train_term_mask, _, train_all_term_mask_fix, train_is_multi = read_data(args.dataset, 'train', args.tokenizer, spc=False)
    dev_sents, dev_terms, dev_labels, dev_term_mask, _, dev_all_term_mask_fix, dev_is_multi = read_data(args.dataset, 'dev', args.tokenizer, spc=False)
    train_sents += dev_sents
    train_labels += dev_labels
    train_terms += dev_terms
    train_term_mask += dev_term_mask
    train_all_term_mask_fix += dev_all_term_mask_fix
    train_is_multi += dev_is_multi
    train_labels = torch.LongTensor(train_labels)
    logger.info('Train Examples: {}'.format(len(train_sents)))
    logger.info('Train Labels:')
    logger.info('Negative: {} | Positive: {} | Netrual: {}'.format(
        torch.sum((train_labels == 0).long()).item(),
        torch.sum((train_labels == 2).long()).item(),
        torch.sum((train_labels == 1).long()).item(),
    ))
    logger.info('--' * 20)

    # Start loop
    loop = tqdm(total=len(train_sents))
    DA_sents = []
    success = 0
    total_count = 0
    # for cur_index in range(len(train_sents)):
    for cur_index in range(10):
        loop.update()

        cur_sent = [args.sos] + train_sents[cur_index] + [args.eos]
        base_sent = [args.pad] * len(cur_sent)
        cur_term = []

        for i in range(len(cur_sent)):
            if train_term_mask[cur_index][i] == 1.:
                base_sent[i] = cur_sent[i]
                cur_term.append(cur_sent[i])

        cur_sent += cur_term + [args.eos]
        base_sent += cur_term + [args.eos]
        train_term_mask[cur_index] += [0] * (len(cur_term) + 1)

        cur_label = train_labels[cur_index]
        logger.info('Current Sent: {}'.format(' '.join([args.tokenizer.convert_tokens_to_string(s).strip() for s in cur_sent])))
        logger.info('Current Label: {}'.format(cur_label))
        logger.info('Current Term: {}'.format(cur_term))

        # Integrated Gradients
        attr = inference(cur_sent, base_sent, args.tokenizer, args.device, cur_label, args.ig)

        # Apply mask
        important_tokens = copy.deepcopy(cur_sent)

        term_mask = train_all_term_mask_fix[cur_index] + [0.] * (len(cur_term) + 1)
        for i in range(len(cur_sent)):
            if term_mask[i] == 1. or cur_sent[i] in [args.sos, args.eos, args.pad]:
                attr[i] = float('-inf')
        sort_attr = copy.deepcopy(attr)
        sort_attr = list(sort_attr.detach().cpu().numpy())
        sort_attr.sort(reverse=True)
        upper_bond = sort_attr[len(sort_attr) // 3]

        for i in range(len(cur_sent)):
            if attr[i] > upper_bond:
                important_tokens[i] = '<mask>'

        # Coutinous mask
        count = 0
        t5_input = args.tokenizer.convert_tokens_to_string(important_tokens[1: len(attr) - len(cur_term) - 2])
        while t5_input.find('<mask><mask>') != -1:
            t5_input = t5_input.replace('<mask><mask>', '<mask>')

        while t5_input.find('<mask>') != -1:
            t5_input = t5_input.replace('<mask>', ' <extra_id_{}>'.format(count), 1)
            count += 1
        t5_input = t5_input.split('</s>')[0]
        if t5_input[:3] == '<s>':
            t5_input = t5_input.replace('<s>', '')

        # Remove signs
        if t5_input[-1] in ['.', ',', ';', ':', '?', '!']:
            t5_input = t5_input[:-1]

        logger.info('T5 input: {}'.format(t5_input))

        pat_type = 'multi' if train_is_multi[cur_index] else 'single'
        cur_term = args.tokenizer.convert_tokens_to_string(cur_term).strip()
        t5_input_list, flip_label_list, pattern_list = add_patten(t5_input, cur_label, pat_type, cur_term)

        # T5 fill in the blank
        with torch.no_grad():
            pred_texts, pred_blanks = t5.generate_blanks(t5_input_list)
        sort_pred_text = []
        for i in range(len(pred_texts)):
            for pre in pred_texts[i]:
                pre = pre.replace(pattern_list[i], ' ' + args.eos + ' ' + cur_term)
                sort_pred_text.append(pre)

        # Find the example with max prob fluctuation
        standard_input = [args.tokenizer.vocab[t] for t in cur_sent]
        standard_input = torch.LongTensor(standard_input).to(args.device)
        cons_input = args.tokenizer(sort_pred_text, return_tensors='pt', padding=True)
        cons_input = cons_input['input_ids'].to(args.device)

        with torch.no_grad():
            standard_output = args.model(standard_input.unsqueeze(0))
            cons_output = args.model(cons_input)
        target_fluctuation = standard_output[0][cur_label] - cons_output[:, cur_label]
        sentiment_labels = [0, 1, 2]
        sentiment_labels.remove(cur_label.item())
        for senti in sentiment_labels:
            target_fluctuation = target_fluctuation + cons_output[:, senti] - standard_output[0][senti]
        max_fluc_index = torch.max(target_fluctuation, dim=-1)[1]
        max_fluc_label = torch.max(cons_output[max_fluc_index], dim=-1)[1]
        max_fluc_value = cons_output[max_fluc_index][max_fluc_label]

        flipped_sent = sort_pred_text[max_fluc_index].replace(pattern_list[max_fluc_index // args.num_return_sequences], '')
        flip_label = flip_label_list[max_fluc_index // args.num_return_sequences]
        logger.info('Flipped Sent: {}'.format(flipped_sent))

        # Define Label
        remain = (max_fluc_label.item() == flip_label) & (max_fluc_value.item() > args.therehold)
        if not remain:
            flip_label = max_fluc_label.item()

        if flip_label != cur_label.item():
            write_sent = flipped_sent + '\t' + str(flip_label) + '\t' + str(1)
        else:
            write_sent = flipped_sent + '\t' + str(flip_label) + '\t' + str(0)

        DA_sents.append(write_sent)

        logger.info('Flipped Label: {}'.format(flip_label))

        logger.info('Standard output: {}'.format(list(standard_output[0].cpu().detach().numpy())))
        logger.info('Flipped output: {}'.format(list(cons_output[max_fluc_index].cpu().detach().numpy())))

        logger.info('--' * 20)

        if flip_label != cur_label:
            success += 1
        total_count += 1
        FlipRate = success / total_count
        loop.set_postfix(FlipRate=FlipRate)

    loop.close()

    # Save
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    open('{}/{}_Counterfactual.txt'.format(args.output_dir, args.dataset), 'w', encoding='utf-8').write('\n'.join(DA_sents))


def setup_everything(args):
    if args.base_model == 'deberta':
        args.sos, args.eos, args.pad = '[CLS]', '[SEP]', '[PAD]'
    elif args.base_model == 'roberta':
        args.sos, args.eos, args.pad = '<s>', '</s>', '<pad>'

    # 读取模型
    model = torch.load(args.model_path, map_location=args.device)
    model.to(args.device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    args.model = model
    args.tokenizer = tokenizer

    def forward_fuc(inputs, token_type_ids=None, position_ids=None, attention_mask=None):
        output = model(inputs)
        return output

    ig = LayerIntegratedGradients(forward_fuc, model.bert.embeddings)

    args.forward_fuc = forward_fuc
    args.ig = ig

    set_seed(args.seed)


if __name__ == "__main__":
    args = parse_args()

    setup_everything(args)

    logger = set_logger(args)

    main(logger, args)
