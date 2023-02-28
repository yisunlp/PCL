import argparse
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset
from model.BertForTokenClassification import BertForTokenClassification
import random
import numpy as np

'''
the utils in RockNER,including preprocessing, args, tags
'''


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda:0", help="set the device id")
    parser.add_argument('--batch_size', type=int, default=16, help="set the batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="set the learning rate")
    parser.add_argument('--seed', type=int, default=2023, help="set the random seed")
    parser.add_argument('--data_path', type=str, help="path to RockNER")
    parser.add_argument('--model_type', type=str, help="model type, should in ['bert-base-cased', 'bert-large-cased]")
    parser.add_argument('--model_path', type=str, help="path of pretrained model, should match model type")
    parser.add_argument('--model_save_path', type=str, required=False, help="path to save the trained model")
    parser.add_argument('--mode', type=str, default="PCL", help="test time mode: PCL, Tent, EATA, OIL")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


args = get_args()
set_seed(args.seed)

# tags in RockNER
TAGS = ['O', 'B-EVENT', 'I-EVENT', 'B-FAC', 'I-FAC', 'B-GPE', 'I-GPE', 'B-LAW', 'B-LOC', 'B-LANGUAGE', 'I-LANGUAGE',
        'I-LAW', 'I-LOC', 'B-NORP', 'I-NORP', 'B-ORG', 'I-ORG', 'B-PERSON', 'B-PRODUCT', 'I-PERSON', 'I-PRODUCT',
        'B-WORK_OF_ART', 'I-WORK_OF_ART']
TAG2IDX = {tag: idx for idx, tag in enumerate(TAGS)}
TAG2IDX['[PAD]'] = -100
IDX2TAG = {idx: tag for idx, tag in enumerate(TAGS)}
IDX2TAG[-100] = '[PAD]'
num_labels = 23
device = torch.device(args.device)
assert args.model_type in ["bert-base-cased", "bert-large-cased"]
if args.model_type == "bert-base-cased":
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
else:
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
model = BertForTokenClassification.from_pretrained(args.model_path, num_labels=num_labels).to(device)
print("using {} as pretrained model".format(args.model_path))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


# prepare the dataset
class NERDataset(Dataset):
    def __init__(self, fpath):
        # dataset is initialized by data path
        self.sentences = []
        self.labels = []
        data = open(fpath, "r", encoding="utf-8").read().strip().split("\n\n")
        for sample in data:
            sample = sample.strip().split("\n")
            self.sentences.append([word.split()[0] for word in sample])
            self.labels.append([word.split()[1] for word in sample])
        assert len(self.sentences) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # add CLS Token and SEP Token
        sentence = ['[CLS]'] + self.sentences[index] + ['[SEP]']
        labels = ['[PAD]'] + self.labels[index] + ['[PAD]']
        input_ids = []
        input_labels = []
        is_heads = []
        for token, label in zip(sentence, labels):
            sub_tokens = tokenizer.tokenize(token) if token not in ('[CLS]', '[SEP]', '[PAD]') else [token]
            is_heads.extend([1] + [0] * (len(sub_tokens) - 1))
            # labels to sub tokens is -100, it should be the same as the ignored index of cross entropy loss
            label = [TAG2IDX[label]] + ([-100] * (len(sub_tokens) - 1))
            input_ids.extend(tokenizer.convert_tokens_to_ids(sub_tokens))
            input_labels.extend(label)
        assert len(input_ids) == len(input_labels) == len(is_heads)
        input_len = len(input_ids)
        is_heads[0], is_heads[-1] = 0, 0
        return input_ids, input_labels, is_heads, input_len, labels[1:-1]


def collate_fn(batch):
    """

    :param batch: a batch of sample
    :return:
            input_ids: input_ids of the samples:[bs, max_len]
            input_labels: labels of the input_ids:[bs, max_len]
            is_valid: whether the corresponding index is valid or not:[bs, max_len]
            labels: labels of the sample
    """
    labels = [sample[4] for sample in batch]
    max_len = max([sample[3] for sample in batch])
    input_ids = [sample[0] + [0] * (max_len - len(sample[0])) for sample in batch]
    input_labels = [sample[1] + [-100] * (max_len - len(sample[1])) for sample in batch]
    is_valid = [sample[2] + [0] * (max_len - len(sample[2])) for sample in batch]
    return torch.tensor(input_ids), torch.tensor(input_labels), is_valid, labels
