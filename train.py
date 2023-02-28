from torch.utils.data import DataLoader
from utils.RockNER_utils import *
from tqdm import tqdm
from seqeval.metrics import f1_score


def train(dataset):
    model.train()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
        input_ids, input_labels = input_ids.to(device), input_labels.to(device)
        outputs = model(input_ids=input_ids, labels=input_labels)
        loss, outputs = outputs[0], outputs[1]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs = outputs.argmax(-1)
        for i in range(outputs.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[outputs[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
    return f1_score(y_true, y_pred)


def test(dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    y_true = []
    y_pred = []
    for input_ids, input_labels, is_heads, labels in tqdm(loader):
        input_ids = input_ids.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids)[0]
        outputs = outputs.argmax(-1)
        for i in range(outputs.size(0)):
            tmp = []
            for j in range(len(is_heads[i])):
                if is_heads[i][j] == 1:
                    tmp.append(IDX2TAG[outputs[i][j].item()])
            if tmp:
                y_pred.append(tmp)
        y_true.extend(labels)
    return f1_score(y_true, y_pred)


def main():
    train_dataset = NERDataset("{}/Original-OntoNotes_train.txt".format(args.data_path))
    dev_full = NERDataset("{}/OntoRock-Full_dev.txt".format(args.data_path))
    dev_context = NERDataset("{}/OntoRock-Context_dev.txt".format(args.data_path))
    dev_entity = NERDataset("{}/OntoRock-Entity_dev.txt".format(args.data_path))
    all_logs = []
    best = 0
    final = []
    for epoch in range(20):
        train_acc = train(train_dataset)
        dev_full_acc = test(dev_full)
        dev_entity_acc = test(dev_entity)
        dev_context_acc = test(dev_context)
        info = "------------------epoch {}---------------------\ntrain_acc:{:.6f}        full:{:.6f}         entity:" \
               "{:.6f}           context:{:.6f}".format(epoch, train_acc, dev_full_acc, dev_entity_acc, dev_context_acc)
        print(info)
        all_logs.append(info)
        if dev_full_acc > best:
            best = dev_full_acc
            model.save_pretrained(args.model_save_path)
            final = [dev_context_acc, dev_entity_acc, dev_full_acc]
    for info in all_logs:
        print(info)
    print("final:   context_acc:{:.6f}  entity_acc:{:.6f}  full_acc:{:.6f}".format(final[0], final[1], final[2]))


main()
