import torch.nn.functional
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score
import copy
import utils.eata as eat
import time
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
    parser.add_argument('--lr', type=float, default=1e-4, help="set the learning rate")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


args = get_args()

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
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


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


def entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1).sum(-1)


def OILCrossEntropy(output, label):
    # compute the OIL cross entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size = output.size()[0]
    output = output.view(-1, 23)
    label = label.view(-1)
    loss = loss_fct(output, label)
    return loss.view(batch_size, -1).mean(-1)


def PCL(dataset):
    model = BertForTokenClassification.from_pretrained("checkpoints/bert-base-rock", num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for name, param in model.named_parameters():
        if "LayerNorm" not in name:
            param.requires_grad = False
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in loader:
        input_ids = input_ids.to(device)
        start = time.time()
        output, loss = model(input_ids=input_ids, perturbation=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output[0].argmax(dim=-1)
        end = time.time()
        eval_time += end - start
        for i in range(output.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[output[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
    return f1_score(y_true, y_pred), eval_time / len(dataset)


def Tent(dataset):
    model = BertForTokenClassification.from_pretrained("checkpoints/bert-base-rock", num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for name, param in model.named_parameters():
        if "LayerNorm" not in name:
            param.requires_grad = False
    model.train()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in loader:
        input_ids = input_ids.to(device)
        start = time.time()
        output = model(input_ids=input_ids)[0]
        loss = torch.mean(entropy(output))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        output = output.argmax(dim=-1)
        end = time.time()
        eval_time += end - start
        for i in range(output.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[output[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
    return f1_score(y_true, y_pred), eval_time / len(dataset)


def EAT(dataset):
    model = BertForTokenClassification.from_pretrained("checkpoints/bert-base-rock", num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for name, param in model.named_parameters():
        if "LayerNorm" not in name:
            param.requires_grad = False
    model.train()
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    adapt_model = eat.EATA(model, optimizer)
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in loader:
        input_ids = input_ids.to(device)
        start = time.time()
        output = adapt_model(input_ids)
        output = output.argmax(dim=-1)
        end = time.time()
        eval_time += end - start
        for i in range(output.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[output[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
    return f1_score(y_true, y_pred), eval_time / len(dataset)


def OIL(dataset):
    model = BertForTokenClassification.from_pretrained("checkpoints/bert-base-rock", num_labels=num_labels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    for name, param in model.named_parameters():
        if "LayerNorm" not in name:
            param.requires_grad = False
    # set OIL hyper_parameters
    # memory_size = args.batch_size  we adapt a batch of sample
    alpha = 0.99  # to update the teacher
    beta = 1  # to do debias
    threshold = 0.5  # to filter the sample
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    y_true = []
    y_pred = []
    loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in loader:
        input_ids = input_ids.to(device)
        start = time.time()
        # do Test
        model.eval()
        with torch.no_grad():
            output = model(input_ids=input_ids)[0]
        with torch.no_grad():
            source_outputs = teacher_model(input_ids=input_ids)[0]
            output = 2 * output - source_outputs - beta * (output - source_outputs)
        output = output.argmax(dim=-1)
        for i in range(output.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[output[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
        # do Train
        model.train()
        with torch.no_grad():
            source_output = teacher_model(input_ids=input_ids)[0]
            pseudo_label = torch.argmax(source_output, dim=-1)
        output = model(input_ids=input_ids)[0]
        loss = OILCrossEntropy(output, copy.deepcopy(pseudo_label))
        loss = torch.sum((loss < threshold) * loss) / torch.sum(loss < threshold)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the source model
        for param1, param2 in zip(teacher_model.parameters(), model.parameters()):
            param1.data = alpha * param1.data + (1 - alpha) * param2.data
        end = time.time()
        eval_time += end - start
    return f1_score(y_true, y_pred), eval_time / len(dataset)


def main():
    # prepare dataset
    dev_full = NERDataset("{}/OntoRock-Full_dev.txt".format("RockNER"))
    dev_context = NERDataset("{}/OntoRock-Context_dev.txt".format("RockNER"))
    dev_entity = NERDataset("{}/OntoRock-Entity_dev.txt".format("RockNER"))
    methods = ["Tent", "EATA", "OIL", "PCL"]
    for method in methods:
        for i in range(20):
            set_seed(i)
            # do Tent
            if method == "Tent":
                context_acc, eval_time1 = Tent(dev_context)
                entity_acc, eval_time2 = Tent(dev_entity)
                full_acc, eval_time3 = Tent(dev_full)
                print("Tent\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(context_acc, entity_acc, full_acc,
                                                                    (context_acc + entity_acc + full_acc) / 3))

            # do EATA
            if method == "EATA":
                context_acc, eval_time1 = EAT(dev_context)
                entity_acc, eval_time2 = EAT(dev_entity)
                full_acc, eval_time3 = EAT(dev_full)
                print("EATA\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(context_acc, entity_acc, full_acc,
                                                                    (context_acc + entity_acc + full_acc) / 3))
            # do OIL
            if method == "OIL":
                context_acc, eval_time1 = OIL(dev_context)
                entity_acc, eval_time2 = OIL(dev_entity)
                full_acc, eval_time3 = OIL(dev_full)
                print("OIL\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(context_acc, entity_acc, full_acc,
                                                                   (context_acc + entity_acc + full_acc) / 3))
            # do PCL
            if method == "PCL":
                context_acc, eval_time1 = PCL(dev_context)
                entity_acc, eval_time2 = PCL(dev_entity)
                full_acc, eval_time3 = PCL(dev_full)
                print("PCL\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}".format(context_acc, entity_acc, full_acc,
                                                                   (context_acc + entity_acc + full_acc) / 3))


main()
