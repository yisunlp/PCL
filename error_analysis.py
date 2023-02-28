import torch.nn.functional
from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score
from utils.RockNER_utils import *
import copy
import utils.eata as eat
import time

for name, param in model.named_parameters():
    if "LayerNorm" not in name:
        param.requires_grad = False


def OILCrossEntropy(output, label):
    # compute the OIL cross entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    batch_size = output.size()[0]
    output = output.view(-1, 23)
    label = label.view(-1)
    loss = loss_fct(output, label)
    return loss.view(batch_size, -1).mean(-1)


def entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1).sum(-1)


def compute_error(y_true, y_pred):
    entitys = []
    true_flag = []
    in_entity = 0
    for sample_id in range(len(y_true)):
        begin_id = 0
        end_id = 0
        for token_id in range(len(y_true[sample_id])):
            tag = y_true[sample_id][token_id]
            tag_type = tag[0]
            if token_id != len(y_true[sample_id]) - 1:
                if in_entity:
                    if tag_type == "B":
                        end_id = token_id
                        entitys.append((sample_id, begin_id, end_id))
                        begin_id = token_id
                    elif tag_type == "I":
                        pass
                    elif tag_type == "O":
                        end_id = token_id
                        entitys.append((sample_id, begin_id, end_id))
                        in_entity = 0
                    else:
                        print("error")
                else:
                    if tag_type == "B":
                        begin_id = token_id
                        in_entity = 1
                    elif tag_type == "O":
                        pass
                    else:
                        print("error")
            else:
                if in_entity:
                    if tag_type == "B":
                        end_id = token_id
                        entitys.append((sample_id, begin_id, end_id))
                        begin_id = token_id
                        entitys.append((sample_id, begin_id, end_id + 1))
                        in_entity = 0
                    elif tag_type == "I":
                        end_id = token_id
                        entitys.append((sample_id, begin_id, end_id + 1))
                        in_entity = 0
                    elif tag_type == "O":
                        end_id = token_id
                        entitys.append((sample_id, begin_id, end_id))
                        in_entity = 0
                    else:
                        print("error")
                else:
                    if tag_type == "B":
                        begin_id = token_id
                        end_id = begin_id + 1
                        entitys.append((sample_id, begin_id, end_id))
                    elif tag_type == "O":
                        pass
                    else:
                        print("error")
    for entity in entitys:
        true_label = y_true[entity[0]][entity[1]:entity[2]]
        pseudo_label = y_pred[entity[0]][entity[1]:entity[2]]
        if true_label == pseudo_label:
            true_flag.append(1)
        else:
            true_flag.append(0)
    return true_flag


def forward(dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
        input_ids = input_ids.to(device)
        start = time.time()
        with torch.no_grad():
            output = model(input_ids=input_ids)[0]
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
    flag = compute_error(y_true, y_pred)
    return flag


def PCL(dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
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
    flag = compute_error(y_true, y_pred)
    return flag


def Tent(dataset):
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
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
    flag = compute_error(y_true, y_pred)
    return flag


def EAT(dataset):
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    adapt_model = eat.EATA(model, optimizer)
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
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
    flag = compute_error(y_true, y_pred)
    return flag


def OIL(dataset):
    # set OIL hyper_parameters
    # memory_size = args.batch_size  we adapt a batch of sample
    alpha = 0.99  # to update the teacher
    beta = 1  # to do debias
    threshold = 0.5  # to filter the sample
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    y_true = []
    y_pred = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    eval_time = 0
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
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
    flag = compute_error(y_true, y_pred)
    return flag


def print_flag(source_flag, target_flag):
    right2right = 0
    right2wrong = 0
    wrong2right = 0
    wrong2wrong = 0
    for i in range(len(source_flag)):
        if source_flag[i] == 1 and target_flag[i] == 1:
            right2right += 1
        if source_flag[i] == 1 and target_flag[i] == 0:
            right2wrong += 1
        if source_flag[i] == 0 and target_flag[i] == 1:
            wrong2right += 1
        if source_flag[i] == 0 and target_flag[i] == 0:
            wrong2wrong += 1
    print("r2r:{}    r2w:{}    w2r:{}    w2w:{}".format(right2right, right2wrong, wrong2right, wrong2wrong))


def main():
    # prepare dataset
    dev_full = NERDataset("{}/OntoRock-Full_dev.txt".format(args.data_path))
    dev_context = NERDataset("{}/OntoRock-Context_dev.txt".format(args.data_path))
    dev_entity = NERDataset("{}/OntoRock-Entity_dev.txt".format(args.data_path))
    base_flag = forward(dev_context) + forward(dev_entity) + forward(dev_full)
    tent_flag = Tent(dev_context) + Tent(dev_entity) + Tent(dev_full)
    eata_flag = EAT(dev_context) + EAT(dev_entity) + EAT(dev_full)
    oil_flag = OIL(dev_context) + OIL(dev_entity) + OIL(dev_full)
    pcl_flag = PCL(dev_context) + PCL(dev_entity) + PCL(dev_full)
    print("Entity nums: {}".format(len(base_flag)))
    print("tent:")
    print_flag(base_flag, tent_flag)
    print("eata:")
    print_flag(base_flag, eata_flag)
    print("OIL:")
    print_flag(base_flag, oil_flag)
    print("PCL:")
    print_flag(base_flag, pcl_flag)


main()
