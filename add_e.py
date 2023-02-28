from torch.utils.data import DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score
from utils.RockNER_utils import *
import time

for name, param in model.named_parameters():
    if "LayerNorm" not in name:
        param.requires_grad = False


def entropy(x):
    return -(x.softmax(-1) * x.log_softmax(-1)).sum(-1).sum(-1)


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


def main():
    dev_full = NERDataset("{}/OntoRock-Full_dev.txt".format(args.data_path))
    dev_context = NERDataset("{}/OntoRock-Context_dev.txt".format(args.data_path))
    dev_entity = NERDataset("{}/OntoRock-Entity_dev.txt".format(args.data_path))
    for i in range(2023,2026):

        set_seed(i)
        model = BertForTokenClassification.from_pretrained(args.model_path, num_labels=num_labels).to(device)
        print("using {} as pretrained model".format(args.model_path))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        for name, param in model.named_parameters():
            if "LayerNorm" not in name:
                param.requires_grad = False
        # prepare dataset
        context_acc, eval_time1 = PCL(dev_context)
        entity_acc, eval_time2 = PCL(dev_entity)
        full_acc, eval_time3 = PCL(dev_full)
        print("{:.6f}\t{:.6f}\t{:.6f}".format(context_acc, entity_acc, full_acc))
main()
