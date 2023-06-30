import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseGCNConv, dense_diff_pool, global_mean_pool,\
    global_add_pool, ChebConv, SAGEConv, GINConv, global_max_pool
import wandb
import torch.nn.functional as F
import numpy as np

global activation_list
activation_list = {}

def get_activation(idx):
    def hook(model, in_put, output):
        if idx == "diff_pool":
            ret_labels = ["pooled_node_feat_matrix", "coarse_adj", "link_pred_loss", "entropy_reg"]
            for l, t in zip(ret_labels, output):
                activation_list[f"{idx}_{l}"] = t.detach()
        else:
            activation_list[idx] = output.detach()
    return hook


def register_hooks(model):
    for name, m in model.named_modules():
            if isinstance(m, GCNConv) or isinstance(m, nn.Linear) or isinstance(m, DenseGCNConv) or isinstance(m, ChebConv) or isinstance(m, SAGEConv) or isinstance(m, GINConv):
                m.register_forward_hook(get_activation(f"{name}"))

    return model

def weights_init(m):
    # if isinstance(m, GCNConv):
    #     torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
    #     torch.nn.init.uniform_(m.bias.data)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("tanh"))
        torch.nn.init.uniform_(m.bias.data)

def test_prediction(model, dataloader, dev):
    correct = 0
    device = torch.device(dev)
    for mod1, mod2, y in dataloader:
        mod1 = mod1.to(device)
        mod2 = mod2.to(device)
        y = y.to(device)
        out = model(mod1, mod2)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())

    return correct / len(dataloader.dataset)
def train_prediction(model, train_loader, test_loader, epochs, lr, num_classes, dev):
    model = register_hooks(model)
    device = torch.device(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss().to(device)

    # list of accuracies
    train_accuracies, test_accuracies, train_loss, test_loss = list(), list(), list(), list()
    train_d = list()

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        num_batches = 0
        for mod1, mod2, y in train_loader:
            mod1 = mod1.to(device)
            mod2 = mod2.to(device)
            y = y.to(device)
            model.train()

            optimizer.zero_grad()

            out = model(mod1, mod2)
            one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).type_as(out)
            loss = criterion(out, one_hot)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            optimizer.step()

        # get accuracy
        train_acc = test_prediction(model, train_loader, dev)
        test_acc = test_prediction(model, test_loader, dev)

        # add to list and print
        model.eval()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # get testing loss
        test_running_loss = 0
        test_num_batches = 0
        for mod1, mod2, y in test_loader:
            mod1 = mod1.to(device)
            mod2 = mod2.to(device)
            y = y.to(device)

            out = model(mod1, mod2)

            one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).type_as(out)

            test_running_loss += criterion(out, one_hot).item()
            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        test_loss.append(test_running_loss / test_num_batches)

        print('Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
            format(epoch, train_loss[-1], test_loss[-1], train_acc, test_acc), end="\r")

        wandb.log({'Epoch': epoch, 'Train loss': train_loss[-1], 'Test Loss': test_loss[-1],
                   'Train Acc': train_acc, 'Test Acc': test_acc})


class Pool(torch.nn.Module):
    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x, batch):
        x = global_mean_pool(x, batch)
        return x

class SumPool(torch.nn.Module):
    def __init__(self):
        super(SumPool, self).__init__()

    def forward(self, x, batch):
        x = global_add_pool(x, batch)
        return x



