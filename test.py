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
    return f1_score(y_true, y_pred), eval_time / len(dataset)


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
    return f1_score(y_true, y_pred), eval_time / len(dataset)


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
    return f1_score(y_true, y_pred), eval_time / len(dataset)


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
    return f1_score(y_true, y_pred), eval_time / len(dataset)


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
    return f1_score(y_true, y_pred), eval_time / len(dataset)


def main():
    # prepare dataset
    dev_full = NERDataset("{}/OntoRock-Full_dev.txt".format(args.data_path))
    dev_context = NERDataset("{}/OntoRock-Context_dev.txt".format(args.data_path))
    dev_entity = NERDataset("{}/OntoRock-Entity_dev.txt".format(args.data_path))
    # do forward
    if args.mode == "forward":
        print("using forward mode")
        context_acc, eval_time1 = forward(dev_context)
        entity_acc, eval_time2 = forward(dev_entity)
        full_acc, eval_time3 = forward(dev_full)
        print("context:{:.6f}     entity:{:.6f}      full:{:.6f}    time:{:.4f}"
              .format(context_acc, entity_acc, full_acc, (eval_time1 + eval_time2 + eval_time3) / 3 * 1000))
    # do PCL
    if args.mode == "PCL":
        print("using PCL mode")
        context_acc, eval_time1 = PCL(dev_context)
        entity_acc, eval_time2 = PCL(dev_entity)
        full_acc, eval_time3 = PCL(dev_full)
        print("context:{:.6f}     entity:{:.6f}      full:{:.6f}    time:{:.4f}"
              .format(context_acc, entity_acc, full_acc, (eval_time1 + eval_time2 + eval_time3) / 3 * 1000))

    # do Tent
    if args.mode == "Tent":
        print("using Tent mode")
        context_acc, eval_time1 = Tent(dev_context)
        entity_acc, eval_time2 = Tent(dev_entity)
        full_acc, eval_time3 = Tent(dev_full)
        print("context:{:.6f}     entity:{:.6f}      full:{:.6f}    time:{:.4f}"
              .format(context_acc, entity_acc, full_acc, (eval_time1 + eval_time2 + eval_time3) / 3 * 1000))

    # do EATA
    if args.mode == "EATA":
        print("using EATA mode")
        context_acc, eval_time1 = EAT(dev_context)
        entity_acc, eval_time2 = EAT(dev_entity)
        full_acc, eval_time3 = EAT(dev_full)
        print("context:{:.6f}     entity:{:.6f}      full:{:.6f}    time:{:.4f}"
              .format(context_acc, entity_acc, full_acc, (eval_time1 + eval_time2 + eval_time3) / 3 * 1000))

    # do OIL
    if args.mode == "OIL":
        print("using OIL mode")
        context_acc, eval_time1 = OIL(dev_context)
        entity_acc, eval_time2 = OIL(dev_entity)
        full_acc, eval_time3 = OIL(dev_full)
        print("context:{:.6f}     entity:{:.6f}      full:{:.6f}    time:{:.4f}"
              .format(context_acc, entity_acc, full_acc, (eval_time1 + eval_time2 + eval_time3) / 3 * 1000))


main()
