import os
import matplotlib.pyplot as plt
import matplotlib

import wandb
import yaml
from tqdm import tqdm

from torch_geometric.data import Data, DataLoader

from torch_geometric.loader import DataLoader

import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything

import clustering_utils
import data_utils
import model_utils
import models
import persistence_utils
import visualisation_utils

import numpy as np
import networkx as nx

import torch
import torch.nn as nn




visualisation_utils.set_rc_params()

def test_graph_class(model, dataloader, if_interpretable_model=True, mode='sharcs'):
    # enter evaluation mode
    correct = 0
    correct_extra = 0
    correct_g = 0
    correct_img = 0
    device = torch.device(dev)
    for data in dataloader:
        data.to(device)
        if if_interpretable_model:
            if mode == 'single_CBM':
                concepts_gnn, concepts_tab, out_g, out_img, c_g, c_image = model(data, data.img, data.img_aux, data.anchor,
                                                                           data.y_g, data.y_img)
                c_g = c_g.argmax(dim=1)
                c_img = c_image.argmax(dim=1)
                y_g = data.y_g
                y_img = data.y_img
                correct_g += int((c_g == y_g).sum())
                correct_img += int((c_img == y_img).sum())
                pred_g = out_g.argmax(dim=1)
                y = data.y
                correct += int((pred_g == y).sum())
                pred_img = out_img.argmax(dim=1)
                y = data.y
                correct_extra += int((pred_img == y).sum())

            else:
                concepts_gnn, concepts_tab, out, out_g, out_img, _ = model(data, data.img, data.img_aux, data.anchor,
                                                                           data.y_g, data.y_img)
                pred_g = out_g.argmax(dim=1)
                pred_img = out_img.argmax(dim=1)
                y_g = data.y_g
                y_img = data.y_img
                correct_g += int((pred_g == y_g).sum())
                correct_img += int((pred_img == y_img).sum())
                pred = out.argmax(dim=1)
                y = data.y
                correct += int((pred == y).sum())

        else:
            if mode == 'vanilla':
                concepts_gnn, concepts_tab, out, out_g, out_img, _ = model(data, data.img, data.img_aux, data.anchor,
                                                                           data.y_g, data.y_img)
                pred = out.argmax(dim=1)
                y = data.y
                correct += int((pred == y).sum())
            elif mode == 'single_vanilla' or mode == 'anchors':
                concepts_gnn, concepts_tab, out, out_g, out_img, _ = model(data, data.img, data.img_aux, data.anchor,
                                                                           data.y_g, data.y_img)
                pred = out_g.argmax(dim=1)
                y = data.y
                correct_g += int((pred == y).sum())
                pred = out_img.argmax(dim=1)
                y = data.y
                correct_img += int((pred == y).sum())
    if if_interpretable_model:
        if mode == 'single_CBM':
            return correct / len(dataloader.dataset), correct_extra / len(dataloader.dataset), \
                correct_g / len(dataloader.dataset), correct_img / len(dataloader.dataset)
        else:
            return correct / len(dataloader.dataset), correct_g / len(dataloader.dataset), correct_img / len(
                dataloader.dataset)
    else:
        if mode == 'vanilla':
            return correct / len(dataloader.dataset), 0, 0
        elif mode == 'single_vanilla' or mode == 'anchors':
            return 0, correct_g / len(dataloader.dataset), correct_img / len(dataloader.dataset)
def train_graph_class(model, train_loader, test_loader, weight, epochs, lr, num_classes,
                      if_interpretable_model=True, mode='sharcs'):
    # register hooks to track activation
    device = torch.device(dev)
    model = model_utils.register_hooks(model)
    optimizer = torch.optim.Adam([
                {'params': model.conv1.parameters(), 'lr': lr*10},
                {'params': model.conv2.parameters(), 'lr': lr*10},
                {'params': model.projection_graph.parameters(), 'lr': lr},
                {'params': model.projection_tab.parameters(), 'lr': lr},
                {'params': model.pred.parameters(), 'lr': lr},
                {'params': model.pred_g.parameters(), 'lr': lr},
                {'params': model.pred_img.parameters(), 'lr': lr},
                {'params': model.conv.parameters(), 'lr': lr}
            ], lr=lr)
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight).to(device)
    criterion2 = nn.BCEWithLogitsLoss().to(device)
    criterion3 = nn.BCELoss().to(device)

    # list of accuracies
    train_accuracies, test_accuracies, train_loss, test_loss = list(), list(), list(), list()
    train_d = list()
    train_img = False
    train_g = False

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        running_dist = 0
        num_batches = 0
        for data in train_loader:
            data.to(device)
            model.train()

            optimizer.zero_grad()

            if if_interpretable_model:
                if mode == 'sharcs' or mode == 'multi_CBM':
                    concepts_gnn, concepts_tab, out, out_g, out_img, d = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img)
                    one_hot_g = torch.nn.functional.one_hot(data.y_g, num_classes=10).type_as(out_g)
                    one_hot_img = torch.nn.functional.one_hot(data.y_img, num_classes=10).type_as(out_img)
                elif mode == 'single_CBM':
                    concepts_gnn, concepts_tab, out_g, out_img, c_g, c_image = model(data, data.img, data.img_aux,
                                                                               data.anchor, data.y_g, data.y_img)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out_g)
                    one_hot_g = torch.nn.functional.one_hot(data.y_g, num_classes=10).type_as(c_g)
                    one_hot_img = torch.nn.functional.one_hot(data.y_img, num_classes=10).type_as(c_image)
                    d = 0
            else:
                concepts_gnn, concepts_tab, out, out_g, out_img, d = model(data, data.img, data.img_aux, data.anchor,
                                                                           data.y_g, data.y_img)
                train_g = True
                train_img = True
            # calculate loss


            # loss = criterion(out, one_hot) + criterion2(out_g, one_hot_g) + criterion2(out_img, one_hot_img) # + 0.005 * d
            if not if_interpretable_model and train_img and train_g and mode == 'vanilla':
                one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out)
                loss = criterion(out, one_hot)
                loss.backward()
                optimizer.step()
            elif not if_interpretable_model and train_img and train_g and mode in ['single_vanilla', 'anchors']:
                one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out_g)
                loss = criterion(out_g, one_hot) + criterion(out_img, one_hot)
                loss.backward()
                optimizer.step()
            elif train_img and train_g:
                if mode == 'single_CBM':
                    loss = criterion(out_g, one_hot) + criterion(out_img, one_hot)
                else:
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out)
                    loss = criterion(out, one_hot) + 0.1 * d
                loss.backward()
                optimizer.step()
            else:
                if mode == 'single_CBM':
                    loss = criterion3(c_g, one_hot_g) + criterion3(c_image, one_hot_img)
                else:
                    loss = criterion2(out_g, one_hot_g) + criterion2(out_img, one_hot_img)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_dist += d
            num_batches += 1

            optimizer.step()

        # get accuracy
        if mode =='single_CBM':
            train_acc, _, _, _ = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model,
                                               mode=mode)
            test_g_acc, test_img_acc, test_c_g, test_c_img = test_graph_class(model, test_loader,
                                                                  if_interpretable_model=if_interpretable_model,
                                                                  mode=mode)
            test_acc = test_g_acc
        else:
            train_acc, _, _ = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            test_acc, test_g_acc, test_img_acc = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model, mode=mode)

        # if (test_img_acc > 0.95 or epoch > 50) and not train_img:
        if (epoch > 15) and not train_img:
            for param in model.conv.parameters():
                param.requires_grad = False
            if mode == 'single_CBM':
                for param in model.projection_tab.parameters():
                    param.requires_grad = False
            elif mode == 'multi_CBM':
                for param in model.projection_tab.parameters():
                    param.requires_grad = False
                for param in model.pred_img.parameters():
                    param.requires_grad = False
            else:
                for param in model.pred_img.parameters():
                    param.requires_grad = False
            train_img = True

        # if (test_g_acc > 0.85 or epoch > 50) and not train_g:
        if (epoch > 15) and not train_g:
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
            if mode == 'single_CBM':
                for param in model.projection_graph.parameters():
                    param.requires_grad = False
            elif mode == 'multi_CBM':
                for param in model.projection_graph.parameters():
                    param.requires_grad = False
                for param in model.pred_g.parameters():
                    param.requires_grad = False
            else:
                for param in model.pred_g.parameters():
                    param.requires_grad = False
            train_g = True

        # add to list and print
        model.eval()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # get testing loss
        test_running_loss = 0
        test_num_batches = 0
        for data in test_loader:
            data.to(device)
            concepts_gnn, concepts_tab, out, out_g, out_img, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img)

            if mode == 'single_vanilla' or mode == 'anchors':
                one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out_g)
                test_running_loss += criterion(out_g, one_hot).item()
            else:
                one_hot = torch.nn.functional.one_hot(data.y, num_classes=num_classes).type_as(out)
                test_running_loss += criterion(out, one_hot).item()

            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        train_d.append(running_dist / num_batches)
        test_loss.append(test_running_loss / test_num_batches)

        if mode == 'single_CBM':
            print('Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test G Acc: {:.5f}, Test Img Acc: {:.5f}, Concept G Acc: {:.5f}, Concept Img Acc: {:.5f}'.
                format(epoch, train_loss[-1], test_loss[-1], train_acc, test_g_acc, test_img_acc, test_c_g, test_c_img))
            wandb.log({'Epoch': epoch, 'Train loss': train_loss[-1], 'Test Loss': test_loss[-1],
                       'Test G Acc': test_g_acc, 'Test Img Acc': test_img_acc, 'Test Concept Graph Acc': test_c_g,
                       'Test Concept Image Acc': test_c_img, 'Dist': train_d[-1]})
        else:
            print('Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}, Test G Acc: {:.5f}, Test Img Acc: {:.5f}, Dist:{:.5f}'.
                      format(epoch, train_loss[-1], test_loss[-1], train_acc, test_acc, test_g_acc, test_img_acc, train_d[-1]))
            wandb.log({'Epoch': epoch, 'Train loss': train_loss[-1], 'Test Loss': test_loss[-1],
                           'Train Acc': train_acc, 'Test Acc': test_acc, 'Test Local Graph Acc': test_g_acc, 'Test Local Image Acc': test_img_acc, 'Dist': train_d[-1]})

    return train_accuracies, test_accuracies, train_loss, test_loss

def print_image(clustering_model, data, x, y, num_nodes_view, all_concepts, path, task='local', concepts=None):
    res_sorted = clustering_utils.get_node_distances(clustering_model, data, concepts)
    sample_graphs = []
    sample_feat = []

    unique_c_unfiltered, counts = np.unique((all_concepts>0.5) + 0, axis=0, return_counts=True)
    counts_filter = counts >= 5
    unique_c = unique_c_unfiltered[counts_filter]
    res_sorted = res_sorted[:, counts_filter]

    l = []

    unique_g_c, counts = np.unique((data>0.5) + 0, axis=0, return_counts=True)
    min_treshold = min(5, len(counts))
    threshold = np.sort(counts)[-min_treshold]
    counts_filter = counts >= threshold
    unique_g_c = unique_g_c[counts_filter].tolist()
    unique_g_c = [''.join(str(e) for e in item) for item in unique_g_c]

    for index, el in enumerate(unique_c):
        tmp = ''.join(str(e) for e in el)
        if tmp in unique_g_c:
            l += [index]

    c = (data>0.5) + 0
    k = len(l)

    if k > 5:
        k = 5
        l = l[:5]
    elif k == 0:
      return sample_graphs, sample_feat

    fig = plt.figure(figsize=(18, 3 * k + 2))
    cols, rows = num_nodes_view, k

    fig.suptitle(f'Nearest Instances to Cluster Centroid for Concepts', y=1.005)

    used_concepts = set()
    top_concepts = []
    top_plot = []
    to_print = []
    for index, i in enumerate(l):
        distances = res_sorted[:, i]

        if num_nodes_view < 0:
            top_indices = np.argsort(distances)[::][num_nodes_view:]
        else:
            top_indices = np.argsort(distances)[::][:num_nodes_view]

        top_instance = x[top_indices]

        labels = y[top_indices]

        concepts_list = c[top_indices]

        raw_concepts = data[top_indices]

        c_tmp = np.array2string((data[int(top_indices[0])]>0.5) + 0, precision=2, separator=',',
                      suppress_small=True)

        if c_tmp not in used_concepts:
            used_concepts.add(c_tmp)
            counter = 0
            for j in range(1, cols+1):
                if counter == 0:
                    top_concepts += [raw_concepts[counter]]
                    to_print += [top_instance[counter].cpu().squeeze()]

                img, label, concept = top_instance[j-1], labels[j-1], concepts_list[j-1]
                fig.add_subplot(rows, cols, index*cols + j)
                plt.title(f'label {label}, concept:{np.where(np.all(unique_c==concept, axis=1))[0]}')
                plt.axis("off")
                # plt.imshow(torch.tensor(img.squeeze().permute(1, 2, 0), dtype=torch.uint8))
                plt.imshow(img.squeeze().cpu(), cmap="gray")
                counter += 1
    fig.tight_layout()
    plt.savefig(os.path.join(path, f"{task}_img_concepts.pdf"))
    plt.savefig(os.path.join(path, f"{task}_img_concepts.png"))
    wandb.log({task: wandb.Image(plt)})
    plt.show()

    for index, el in enumerate(to_print):
        fig_aux, ax_aux = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))  # create figure & 1 axis
        plt.imshow(el.squeeze().cpu(), cmap="gray")
        plt.axis("off")
        plt.savefig(f"./images/half_{index}_image.png")
        im = matplotlib.image.imread(f"./images/half_{index}_image.png")
        top_plot += [im]


    return top_plot, top_concepts

def plot_samples(clustering_model, data, x, pos_g, batch, y, k, num_nodes_view, edges, all_concepts, path, task='local', concepts=None):
    res_sorted = clustering_utils.get_node_distances(clustering_model, data, concepts)
    sample_graphs = []
    sample_feat = []

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    unique_c_unfiltered, counts = np.unique((all_concepts>0.5) + 0, axis=0, return_counts=True)
    counts_filter = counts >= 5
    unique_c = unique_c_unfiltered[counts_filter]
    res_sorted = res_sorted[:, counts_filter]

    l = []

    unique_g_c, counts = np.unique((data>0.5) + 0, axis=0, return_counts=True)
    min_treshold = min(5, len(counts))
    threshold = np.sort(counts)[-min_treshold]
    counts_filter = counts >= threshold
    unique_g_c = unique_g_c[counts_filter].tolist()
    unique_g_c = [''.join(str(e) for e in item) for item in unique_g_c]

    for index, el in enumerate(unique_c):
        tmp = ''.join(str(e) for e in el)
        if tmp in unique_g_c:
            l += [index]

    c = (data>0.5) + 0
    k = len(l)

    if k > 5:
        k = 5
        l = l[:5]
    elif k == 0:
      return sample_graphs, sample_feat

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to Cluster Centroid for Concepts', y=1.005)
    top_concepts = []
    top_plot = []
    for i, ax_list in zip(l, axes):

        distances = res_sorted[:, i]

        top_graphs, color_maps, pos_maps = [], [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            raw_concepts = data[top_indices]
            tg, cm, labels, concepts_list = clustering_utils.get_top_graphs(top_indices, c, y, edges, batch)


            node_emb = [(x[list(t)]/255).detach().cpu().numpy() for t in tg]
            node_pos = [(pos_g[list(t)]).detach().cpu().numpy() for t in tg]
            for i, el in enumerate(node_pos):
                node_pos[i][:, 1] = np.absolute(np.subtract(node_pos[i][:, 1], np.amax(node_pos[i], axis=0)[1]))
            node_pos = [dict(zip(list(t), pos)) for t, pos in zip(tg, node_pos)]
            top_graphs = top_graphs + tg
            color_maps = color_maps + node_emb
            pos_maps = pos_maps + node_pos

        if k == 1:
            ax_list = [ax_list]
        counter = 0
        for ax, new_G, color_map, g_label, g_concept, pos, g_raw in zip(ax_list, top_graphs, color_maps, labels, concepts_list, node_pos, raw_concepts):
            if counter == 0:
                top_concepts += [g_raw]
                fig_aux = plt.figure(figsize=(6, 3))
                nx.draw(new_G, node_color=color_map, pos=pos, ax=fig_aux.add_subplot(), node_size=250)
                plt.axis('off')
                fig_aux.savefig(f"./images/half_{i}_graph.png")
                im = matplotlib.image.imread(f"./images/half_{i}_graph.png")
                top_plot += [im]
            nx.draw(new_G, node_color=color_map, pos=pos, ax=ax)
            ax.set_title(f"label {g_label}, concept {np.where(np.all(unique_c==g_concept, axis=1))[0]}", fontsize=14)
            ax.axis('off')
            counter += 1

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])
    fig.tight_layout()
    plt.savefig(os.path.join(path, f"{task}_g_concepts.pdf"))
    plt.savefig(os.path.join(path, f"{task}_g_concepts.png"))
    wandb.log({task: wandb.Image(plt)})
    plt.show()

    return top_plot, top_concepts


def print_near_example(t_concepts, x_tab, g_concepts, expanded_x, expanded_pos, edges, batch, y_img, y_g, path, times=2, example=4):
    d = nn.PairwiseDistance(p=2)
    figure = plt.figure(figsize=(18, 3 * times + 2))
    image_indexes = []
    graph_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(t_concepts), size=(1,)).item()
        dist = d(torch.Tensor(g_concepts), torch.Tensor(t_concepts[sample_idx])).squeeze(-1)
        g_index = torch.argsort(dist)[:example]
        image_indexes += [sample_idx]
        graph_indexes += [list(g_index.numpy())]
    cols, rows = 5, times
    edges = torch.transpose(edges, 1, 0).cpu()
    df = pd.DataFrame(edges.numpy())
    for i in range(1, times + 1):
        sample_idx = image_indexes[i-1]
        img, label = x_tab[sample_idx], y_img[sample_idx]
        figure.add_subplot(rows, cols, (i-1)*(example+1)+1)
        plt.title(f'label {label.item()}', fontsize=25)
        plt.axis("off")
        # plt.imshow(torch.tensor(img.squeeze().permute(1, 2, 0), dtype=torch.uint8))
        plt.imshow(img.squeeze().cpu(), cmap="gray")
        for j, idx in enumerate(graph_indexes[i-1]):
            node_indexes = torch.Tensor(list(range(batch.shape[0])))
            neighbours = node_indexes[batch == idx].numpy()
            neighbours = list(set(neighbours))
            df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
            edges = df_neighbours.to_numpy()
            label = y_g[idx]
            new_G = nx.Graph()
            new_G.add_edges_from(edges)
            node_emb = (expanded_x[list(new_G)] / 255).detach().cpu().numpy()
            node_pos = expanded_pos[list(new_G)].detach().cpu().numpy()
            node_pos[:, 1] = np.absolute(np.subtract(node_pos[:, 1], np.amax(node_pos, axis=0)[1]))
            node_pos = dict(zip(list(new_G), node_pos))
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            nx.draw(new_G, node_color=node_emb, pos=node_pos)
            plt.title(f"label {label}", fontsize=25)
            plt.axis('off')
    figure.tight_layout()
    plt.savefig(os.path.join(path, f"similar_img.pdf"))
    plt.savefig(os.path.join(path, f"similar_img.png"))
    wandb.log({'similar_img': wandb.Image(plt)})
    plt.show()

    figure = plt.figure(figsize=(18, 3 * times + 2))
    image_indexes = []
    graph_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(t_concepts), size=(1,)).item()
        dist = d(torch.Tensor(t_concepts), torch.Tensor(g_concepts[sample_idx])).squeeze(-1)
        im_index = torch.argsort(dist)[:example]
        image_indexes += [list(im_index.numpy())]
        graph_indexes += [sample_idx]
    cols, rows = 5, times
    for i in range(1, times + 1):
        sample_idx = graph_indexes[i - 1]
        node_indexes = torch.Tensor(list(range(batch.shape[0])))
        neighbours = node_indexes[batch == sample_idx].numpy()
        neighbours = list(set(neighbours))
        df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
        edges = df_neighbours.to_numpy()
        label = y_g[sample_idx]
        new_G = nx.Graph()
        new_G.add_edges_from(edges)
        node_emb = (expanded_x[list(new_G)] / 255).detach().cpu().numpy()
        node_pos = expanded_pos[list(new_G)].detach().cpu().numpy()
        node_pos[:, 1] = np.absolute(np.subtract(node_pos[:, 1], np.amax(node_pos, axis=0)[1]))
        node_pos = dict(zip(list(new_G), node_pos))
        figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 1)
        nx.draw(new_G, node_color=node_emb, pos=node_pos)
        plt.title(f"label {label}", fontsize=25)
        plt.axis('off')
        for j, idx in enumerate(image_indexes[i - 1]):
            img, label = x_tab[idx], y_img[idx]
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            plt.title(f'label {label.item()}', fontsize=25)
            plt.axis("off")
            # plt.imshow(torch.tensor(img.squeeze().permute(1, 2, 0), dtype=torch.uint8))
            plt.imshow(img.squeeze().cpu(), cmap="gray")
    figure.tight_layout()
    figure.tight_layout()
    plt.savefig(os.path.join(path, f"similar_g.pdf"))
    plt.savefig(os.path.join(path, f"similar_g.png"))
    wandb.log({'similar_img': wandb.Image(plt)})
    plt.show()

def retreived_similar(mod, similar_concepts):
    d = nn.PairwiseDistance(p=2)
    similar_concepts = torch.Tensor(similar_concepts)
    retreived_idx = []
    for i in range(mod.shape[0]):
        dist = d(mod[i], similar_concepts).squeeze(-1)
        g_index = torch.argsort(dist)[0]
        retreived_idx += [g_index.item()]
    retreived = similar_concepts[retreived_idx]
    return retreived

def test_missing_modality(dataloader, model, concepts_mod1, concepts_mod2):
    device = torch.device(dev)
    correct = 0
    for data in dataloader:
        data.to(device)
        tab, tab_aux = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, mod1=True)
        retreived_graph = retreived_similar(tab_aux.detach().cpu(), concepts_mod1)
        out = model(retreived_graph.to(device), tab, data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, prediction=True)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for data in dataloader:
        data.to(device)
        data_img = Data(x=data.x_image, edge_index=data.edge_index_image,
                        edge_attr=data.edge_attr_image, batch=data.x_image_batch, pos=data.pos_image)
        graph, graph_aux = model(data, data_img, data.img_aux, data.anchor, data.y_g, data.y_img,
                                 missing=True, mod2=True)
        retreived_image = retreived_similar(graph_aux.detach().cpu(), concepts_mod2)
        out = model(graph, retreived_image.to(device), data.img_aux, data.anchor, data.y_g, data.y_img,
                    missing=True, prediction=True)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc2 = correct / len(dataloader.dataset)
    return acc1, acc2

def test_missing_modality_anchors(dataloader, model, pred_model, concepts_mod1, concepts_mod2, anchors_mod1, anchors_mod2):
    device = torch.device(dev)
    correct = 0
    print('----MISSING MOD 1----')
    for data in dataloader:
        data.to(device)
        tab, tab_aux = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, mod1=True)
        tab_aux = compute_relative_rep(tab_aux.detach().cpu(), anchors_mod2)
        tab = compute_relative_rep(tab.detach().cpu(), anchors_mod2)
        retreived_graph = retreived_similar(tab_aux, concepts_mod1)
        out = pred_model(tab.to(device), retreived_graph.to(device))
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for data in dataloader:
        data.to(device)
        data_img = Data(x=data.x_image, edge_index=data.edge_index_image,
                        edge_attr=data.edge_attr_image, batch=data.x_image_batch, pos=data.pos_image)
        graph, graph_aux = model(data, data_img, data.img_aux, data.anchor, data.y_g, data.y_img,
                                 missing=True, mod2=True)
        graph_aux = compute_relative_rep(graph_aux.detach().cpu(), anchors_mod1)
        graph = compute_relative_rep(graph.detach().cpu(), anchors_mod1)
        retreived_tab = retreived_similar(graph_aux, concepts_mod2)
        out = pred_model(retreived_tab.to(device), graph.to(device))
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc2 = correct / len(dataloader.dataset)
    return acc1, acc2

def collect_hidden_representation(model, dataloader):
    model.eval()
    device = torch.device(dev)
    img = []
    graph = []
    y = []
    for data in dataloader:
        data.to(device)
        tmp_graph, tmp_image, _, _, _, _ = model(data, data.img, data.img_aux, data.anchor,
                                                                   data.y_g, data.y_img)
        img += [tmp_image.detach().cpu()]
        graph += [tmp_graph.detach().cpu()]
        y += [data.y.detach().cpu()]
    img = torch.vstack(img)
    graph = torch.vstack(graph)
    y = torch.hstack(y)
    return img, graph, y

def choose_anchors(model, dataloader):
    data = next(iter(dataloader))
    device = torch.device(dev)
    data.to(device)

    tmp_graph, tmp_image, _, _, _, _ = model(data, data.img_aux, data.img_aux, data.anchor,
                                             data.y_g, data.y_img)
    return tmp_image.detach().cpu(), tmp_graph.detach().cpu()

def compute_relative_rep(mod, anchors):
        result = torch.Tensor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in anchors:
            tmp = cos(mod, t).unsqueeze(dim=-1)
            result = torch.cat((result, tmp), dim=-1)
        return result

def save_centroids(centroids, y, used_centroid_labels, union_concepts,
                           g_concepts, edges, batch, x, pos_g,
                           img_concepts, images,
                           path):

    res_sorted = clustering_utils.get_node_distances(None, union_concepts, centroids)
    unique_concepts = np.unique(used_centroid_labels)
    g_con = (g_concepts > 0.5) + 0
    for c in tqdm(unique_concepts):
        distances = res_sorted[:, c]
        top_indices = np.argsort(distances)[::][0]
        img = images[top_indices]

        tg, cm, labels, concepts_list = clustering_utils.get_top_graphs([top_indices], g_con, y, edges, batch)
        fig = plt.figure(figsize=(3, 3))
        fig.add_subplot(1, 2, 1)
        node_emb = [(x[list(t)] / 255).detach().cpu().numpy() for t in tg]
        node_pos = [(pos_g[list(t)]).detach().cpu().numpy() for t in tg]
        for i, el in enumerate(node_pos):
            node_pos[i][:, 1] = np.absolute(np.subtract(node_pos[i][:, 1], np.amax(node_pos[i], axis=0)[1]))
        node_pos = [dict(zip(list(t), pos)) for t, pos in zip(tg, node_pos)]
        nx.draw(tg[0], node_color=node_emb[0], pos=node_pos[0])
        fig.add_subplot(1, 2, 2)
        plt.imshow(img.squeeze().cpu(), cmap="gray")
        plt.axis('off')
        plt.savefig(f'{path}/{c}.svg')

def test_with_incremental_noise(model, dataloader, if_interpretable_model=True, mode='sharcs'):
    correct = 0
    correct_t = 0
    device = torch.device(dev)
    for data in dataloader:
        data.to(device)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn, concepts_tab, out, _, _, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, noise="mod1")
                concepts_gnn, concepts_tab, out2, _, _, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, noise="mod2")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        pred_t = out2.argmax(dim=1)
        correct_t += int((pred_t == data.y).sum())

    return correct / len(dataloader.dataset), correct_t / len(dataloader.dataset)

def test_with_interventions(model, n_concepts, dataloader, concepts_mod1, concepts_mod2, if_interpretable_model=True, mode='sharcs'):
    device = torch.device(dev)   
    correct = 0
    correct_m= 0
    correct2 = 0
    correct_m2= 0
    p = int(n_concepts/2)
    for data in dataloader:
        data.to(device)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn_noise, concepts_tab_noise, out, _, _, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, noise="mod1")
                concepts_gnn_noise2, concepts_tab_noise2, out2, _, _, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, noise="mod2")
                concepts_gnn, concepts_tab, out, _, _, _ = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img)
                concepts_noise = torch.cat([concepts_gnn_noise, concepts_tab_noise], dim=-1)
                concepts_noise2 = torch.cat([concepts_gnn_noise2, concepts_tab_noise2], dim=-1)
                concepts = torch.cat([concepts_gnn, concepts_tab], dim=-1)
                
                tab, tab_aux = model(data, data.img, data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, mod1=True)
                retreived_graph = retreived_similar(tab_aux.detach().cpu(), concepts_mod1).to(device)
                concepts_retrieved1 = torch.cat([retreived_graph, tab], dim=-1)
                concepts_noise_missing_mod1 = concepts_noise.clone()
                
                data_img = Data(x=data.x_image, edge_index=data.edge_index_image,
                        edge_attr=data.edge_attr_image, batch=data.x_image_batch, pos=data.pos_image)
                graph, graph_aux = model(data, data_img, data.img_aux, data.anchor, data.y_g, data.y_img,
                                 missing=True, mod2=True)
                retreived_tab = retreived_similar(graph_aux.detach().cpu(), concepts_mod2).to(device)
                concepts_retrieved2 = torch.cat([graph, retreived_tab], dim=-1)
                concepts_noise_missing_mod2 = concepts_noise2.clone()
                
                # change p random values in concepts_noise according to concepts
                c = np.random.choice(n_concepts, p, replace=False)
                for i in c:
                    concepts_noise[:, i] = concepts[:, i]
                    concepts_noise_missing_mod1[:, i] = concepts_retrieved1[:, i]
                    concepts_noise2[:, i+n_concepts] = concepts[:, i+n_concepts]
                    concepts_noise_missing_mod2[:, i+n_concepts] = concepts_retrieved2[:, i+n_concepts]
                    
                g_concepts = concepts_noise[:, :int(concepts_noise.shape[1] / 2)]
                t_concepts = concepts_noise[:, int(concepts_noise.shape[1] / 2):]
                out = model(g_concepts.to(device), t_concepts.to(device), data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, prediction=True)
                g_concepts = concepts_noise2[:, :int(concepts_noise2.shape[1] / 2)]
                t_concepts = concepts_noise2[:, int(concepts_noise2.shape[1] / 2):]
                out2 = model(g_concepts.to(device), t_concepts.to(device), data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, prediction=True)
            
                g_concepts_missing_mod = concepts_noise_missing_mod1[:, :int(concepts_noise_missing_mod1.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod1[:, int(concepts_noise_missing_mod1.shape[1] / 2):]
                out_missing_mod = model(g_concepts_missing_mod.to(device), t_concepts_missing_mod.to(device), data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, prediction=True)
                g_concepts_missing_mod = concepts_noise_missing_mod2[:, :int(concepts_noise_missing_mod2.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod2[:, int(concepts_noise_missing_mod2.shape[1] / 2):]
                out_missing_mod2 = model(g_concepts_missing_mod.to(device), t_concepts_missing_mod.to(device), data.img_aux, data.anchor, data.y_g, data.y_img, missing=True, prediction=True)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
        pred_missing = out_missing_mod.argmax(dim=1)
        correct_m += int((pred_missing == data.y).sum())

        pred2 = out2.argmax(dim=1)
        correct2 += int((pred2 == data.y).sum())
        pred_missing2 = out_missing_mod2.argmax(dim=1)
        correct_m2 += int((pred_missing2 == data.y).sum())

    return correct / len(dataloader.dataset), correct_m / len(dataloader.dataset), correct2 / len(dataloader.dataset), correct_m2 / len(dataloader.dataset)


def main():
    tag = 'halfmnist'
    with open(f'./config/{tag}_sweep.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open(f'./config/{tag}.yaml') as file:
        config_param = yaml.load(file, Loader=yaml.FullLoader)
    config = {**config_param, **config}

    run = wandb.init(tags=tag, config=config)

    seed = wandb.config.seed
    seed_everything(seed)

    DATASET_NAME = wandb.config.dataset['values']

    # constants
    MODE = wandb.config.mode

    path = os.path.join("output", DATASET_NAME, MODE, f"seed_{seed}")
    data_utils.create_path(path)

    MODEL_NAME = f"{DATASET_NAME}_{MODE}"
    LOCAL_NUM_CLASSES = wandb.config.local_num_classes['values']
    NUM_CLASSES = wandb.config.num_classes['values']
    SIZE = wandb.config.size['values']

    TRAIN_TEST_SPLIT = 0.8

    NUM_HIDDEN_UNITS = wandb.config.hidden_dim['values']
    CLUSTER_ENCODING_SIZE = wandb.config.k['values']
    EPOCHS = wandb.config.epochs['values']

    LR = wandb.config.lr['values']

    BATCH_SIZE = wandb.config.batch_size['values']

    global dev
    dev = wandb.config.dev['values']

    LAYER_NUM = 3

    config = {'seed': seed,
                       'dataset_name': DATASET_NAME,
                       'model_name': MODEL_NAME,
                       'num_classes': NUM_CLASSES,
                       'train_test_split': TRAIN_TEST_SPLIT,
                       'batch_size': BATCH_SIZE,
                       'num_hidden_units': NUM_HIDDEN_UNITS,
                       'cluster_encoding_size': CLUSTER_ENCODING_SIZE,
                       'epochs': EPOCHS,
                       'lr': LR,
                      }
    persistence_utils.persist_experiment(config, path, 'config.z')

    # load data
    print("Reading dataset")


    # dataset = load_dataset("svhn", 'cropped_digits')
    train_loader, train, test_loader, test, weights = data_utils.create_halfmnist(SIZE, BATCH_SIZE, NUM_CLASSES)

    full_train_loader = DataLoader(train, batch_size=int(len(train) * 0.1), shuffle=True,  follow_batch=['x', 'x_image'])
    full_test_loader = DataLoader(test, batch_size=int(len(test) * 0.1), shuffle=True,  follow_batch=['x', 'x_image'])
    print('Done!')

    # model training
    if MODE == 'vanilla':
        model = models.HalfMnist_vanilla(CLUSTER_ENCODING_SIZE, LOCAL_NUM_CLASSES, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_vanilla' or MODE == 'anchors':
        model = models.HalfMnist_vanilla_single(CLUSTER_ENCODING_SIZE, LOCAL_NUM_CLASSES, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_CBM':
        model = models.HalfMnist_cbm_single(CLUSTER_ENCODING_SIZE, LOCAL_NUM_CLASSES, NUM_CLASSES)
        interpretable = True
    elif MODE == 'multi_CBM':
        model = models.HalfMnist_MultiCBM(CLUSTER_ENCODING_SIZE, LOCAL_NUM_CLASSES, NUM_CLASSES)
        interpretable = True
    else:
        model = models.HalfMnist_missing(CLUSTER_ENCODING_SIZE, LOCAL_NUM_CLASSES, NUM_CLASSES)
        interpretable = True
    model.to(dev)

    model_to_return = model
    # register hooks to track activation

    model = model_utils.register_hooks(model)

    # train
    train_acc, test_acc, train_loss, test_loss = train_graph_class(model, train_loader, test_loader, weights, EPOCHS, LR, NUM_CLASSES,
                                                                   if_interpretable_model=interpretable, mode=MODE)
    persistence_utils.persist_model(model, path, 'model.z')
    if MODE == 'anchors':
        train_tab_representation, train_graph_representation, train_y = collect_hidden_representation(model,
                                                                                                      train_loader)
        test_tab_representation, test_graph_representation, test_y = collect_hidden_representation(model, test_loader)

        train_loader = DataLoader(train, batch_size=int(len(train) * 0.1), shuffle=True,
                                  follow_batch=['x', 'x_image'])
        anchors_tab, anchors_graph = choose_anchors(model, train_loader)
        train_relative_tab = compute_relative_rep(train_tab_representation, anchors_tab)
        train_relative_graph = compute_relative_rep(train_graph_representation, anchors_graph)
        test_relative_tab = compute_relative_rep(test_tab_representation, anchors_tab)
        test_relative_graph = compute_relative_rep(test_graph_representation, anchors_graph)

        train_set = data_utils.Anchors(train_relative_tab, train_relative_graph, train_y)
        test_set = data_utils.Anchors(test_relative_tab, test_relative_graph, test_y)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

        input_size = train_set[0][0].shape[0] + train_set[0][1].shape[0]

        pred_model = models.PredModel(input_size, input_size * 3, NUM_CLASSES)
        pred_model.to(dev)

        model_utils.train_prediction(pred_model, train_loader, test_loader, 20, LR, NUM_CLASSES, dev)

        print("\n_____________THIS IS FOR GRAPHS AND TABLE____________")

        train_data = next(iter(full_train_loader)).to(dev)
        test_data = next(iter(full_test_loader)).to(dev)

        x_train = train_data.x.cpu()
        x_test = test_data.x.cpu()
        expanded_x = torch.cat((x_train, x_test))

        train_graph, train_tab, _, _, _, _ = model(train_data, train_data.img, train_data.img_aux, train_data.anchor, train_data.y_g, train_data.y_img)
        train_graph = train_graph.cpu()
        train_tab = train_tab.cpu()

        train_tab = compute_relative_rep(train_tab, anchors_tab)
        train_graph = compute_relative_rep(train_graph, anchors_graph)

        test_graph, test_tab, _, _, _, _ = model(test_data, test_data.img, test_data.img_aux, test_data.anchor, test_data.y_g, test_data.y_img)

        test_graph = test_graph.cpu()
        test_tab = test_tab.cpu()

        test_tab = compute_relative_rep(test_tab, anchors_tab)
        test_graph = compute_relative_rep(test_graph, anchors_graph)

        graph_concepts = torch.vstack([train_graph, test_graph])
        tab_concepts = torch.vstack([train_tab, test_tab])

        pos_train = train_data.pos.cpu()
        pos_test = test_data.pos.cpu()
        expanded_pos = torch.cat((pos_train, pos_test))

        edges_train = train_data.edge_index.cpu()
        edges_test = test_data.edge_index.cpu()
        offset = train_data.x.shape[0]
        edges_test = edges_test + offset
        expanded_edges = torch.cat((edges_train, edges_test), dim=-1)

        y_train = train_data.y.cpu()
        y_test = test_data.y.cpu()
        y = torch.cat((y_train, y_test))

        y_img_train = train_data.y_img.cpu()
        y_img_test = test_data.y_img.cpu()
        y_img = torch.cat((y_img_train, y_img_test))

        y_g_train = train_data.y_g.cpu()
        y_g_test = test_data.y_g.cpu()
        y_g = torch.cat((y_g_train, y_g_test))

        offset = train_data.batch[-1].cpu() + 1
        batch = torch.cat((train_data.batch.cpu(), test_data.batch.cpu() + offset))

        concepts = torch.vstack([torch.Tensor(graph_concepts), torch.Tensor(tab_concepts)]).detach()
        y_double = torch.cat((y, y), dim=0)
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts, y_double)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels.z')

        # plot clustering
        tab_or_graph = torch.cat((torch.ones(int(concepts.shape[0] / 2)), torch.zeros(int(concepts.shape[0] / 2))),
                                 dim=0)
        visualisation_utils.plot_clustering(seed, concepts, tab_or_graph, centroids, centroid_labels,
                                            used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="shared",
                                            id_path="_graph")

        edges_train = train_data.edge_index.transpose(0, 1).detach().cpu().numpy()
        edges_test = test_data.edge_index.transpose(0, 1).detach().cpu().numpy() + train_data.x.shape[0]

        edges_t = np.concatenate((edges_train, edges_test), axis=0)
        expanded_batch = np.concatenate(
            (train_data.batch.cpu().numpy(), test_data.batch.cpu().numpy() + train_data.batch[-1].item() + 1), axis=0)
        g_concepts = graph_concepts.detach().cpu().numpy()

        t_concepts = tab_concepts.detach().numpy()
        train_img = train_data.img
        test_img = test_data.img
        x_tab = torch.cat((train_img, test_img))

        print('-------- SIMILAR EXAMPLE FROM DIFFERENT MODALITIES----------')
        print_near_example(t_concepts, x_tab, g_concepts, expanded_x, expanded_pos, expanded_edges, expanded_batch,
                           y_img, y_g, path)

        print('----------MISSING MODALITY------------')
        missing_accuracy = test_missing_modality_anchors(full_test_loader, model, pred_model,
                                                         g_concepts, t_concepts,
                                                         anchors_graph, anchors_tab)
        wandb.log({'missing graph modality accuracy': missing_accuracy[0],
                   'missing tab modality accuracy': missing_accuracy[1]})
        print(f'Missing accuracy: {missing_accuracy}')

    if interpretable:
        # get model activations for complete dataset
        train_data = next(iter(full_train_loader)).to(dev)
        test_data = next(iter(full_test_loader)).to(dev)

        x_train = train_data.x.cpu()
        x_test = test_data.x.cpu()
        expanded_x = torch.cat((x_train, x_test))

        train_node_concepts, _, _, _, _, _ = model(train_data, train_data.img, train_data.img_aux, train_data.anchor, train_data.y_g, train_data.y_img)
        train_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        train_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        train_tab_local_concepts = model.x_image_local_concepts.cpu()
        train_tab_concepts = model.x_image_shared_concepts.cpu()

        test_node_concepts, _, _, _, _, _ = model(test_data, test_data.img, test_data.img_aux, test_data.anchor, test_data.y_g, test_data.y_img)
        test_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        test_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        test_tab_local_concepts = model.x_image_local_concepts.cpu()
        test_tab_concepts = model.x_image_shared_concepts.cpu()

        pos_train = train_data.pos.cpu()
        pos_test = test_data.pos.cpu()
        expanded_pos = torch.cat((pos_train, pos_test))

        edges_train = train_data.edge_index.cpu()
        edges_test = test_data.edge_index.cpu()
        offset = train_data.x.shape[0]
        edges_test = edges_test + offset
        expanded_edges = torch.cat((edges_train, edges_test), dim=-1)

        node_concepts = torch.vstack((train_node_concepts, test_node_concepts))
        graph_concepts = torch.vstack([train_graph_concepts, test_graph_concepts])
        graph_local_concepts = torch.vstack([train_graph_local_concepts, test_graph_local_concepts])
        img_local_concepts = torch.vstack([train_tab_local_concepts, test_tab_local_concepts])
        tab_concepts = torch.vstack([train_tab_concepts, test_tab_concepts])

        y_train = train_data.y.cpu()
        y_test = test_data.y.cpu()
        y = torch.cat((y_train, y_test))

        y_img_train = train_data.y_img.cpu()
        y_img_test = test_data.y_img.cpu()
        y_img = torch.cat((y_img_train, y_img_test))

        y_g_train = train_data.y_g.cpu()
        y_g_test = test_data.y_g.cpu()
        y_g = torch.cat((y_g_train, y_g_test))

        train_mask = np.zeros(y.shape[0], dtype=bool)
        train_mask[:train_data.y.shape[0]] = True
        test_mask = ~train_mask

        offset = train_data.batch[-1].cpu() + 1
        batch = torch.cat((train_data.batch.cpu(), test_data.batch.cpu() + offset))

        persistence_utils.persist_experiment(node_concepts, path, 'node_concepts.z')
        persistence_utils.persist_experiment(graph_concepts, path, 'graph_concepts.z')
        persistence_utils.persist_experiment(tab_concepts, path, 'tab_concepts.z')

        print("\n_____________THIS IS FOR GRAPHS____________")
        concepts_g_local = torch.Tensor(graph_local_concepts).detach()
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts_g_local, y_g)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids_g.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels_g.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels_g.z')

        # calculate cluster sizing
        cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
        concepts_g_local_tree = (concepts_g_local.detach().numpy() > 0.5).astype(int)
        classifier = models.ActivationClassifierConcepts(y_g, concepts_g_local_tree, train_mask, test_mask)
        print(f"Classifier Concept completeness score: {classifier.accuracy}")
        concept_metrics = [('cluster_count', cluster_counts)]
        persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
        wandb.log({'local graph completeness': classifier.accuracy, 'num clusters local graph': len(centroids)})

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_g_local, y_g, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="graph local", id_path="_graph")

        edges_train = train_data.edge_index.transpose(0, 1).detach().cpu().numpy()
        edges_test = test_data.edge_index.transpose(0, 1).detach().cpu().numpy() + train_data.x.shape[0]

        edges_t = np.concatenate((edges_train, edges_test), axis=0)
        expanded_batch = np.concatenate((train_data.batch.cpu().numpy(), test_data.batch.cpu().numpy() + train_data.batch[-1].item() + 1), axis=0)
        g_concepts = concepts_g_local.detach().cpu().numpy()
        print('GRAPH CONCEPTS')
        sample_graphs, sample_feat = plot_samples(None, g_concepts, expanded_x, expanded_pos, expanded_batch, y_g, len(centroids), 5, edges_t, concepts_g_local, path, task='local', concepts=centroids)

        print("\n_____________THIS IS FOR IMAGE____________")
        concepts_img_local = torch.Tensor(img_local_concepts).detach()
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts_img_local, y_img)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids_img.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels_img.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels_img.z')

        # calculate cluster sizing
        cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
        concepts_img_local_tree = (concepts_img_local.detach().numpy() > 0.5).astype(int)
        classifier = models.ActivationClassifierConcepts(y_img, concepts_img_local_tree, train_mask, test_mask)

        print(f"Classifier Concept completeness score: {classifier.accuracy}")
        concept_metrics = [('cluster_count', cluster_counts)]
        persistence_utils.persist_experiment(concept_metrics, path, 'image_concept_metrics.z')
        wandb.log({'local image completeness': classifier.accuracy, 'num clusters local graph': len(centroids)})

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_img_local, y_img, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="image local", id_path="_graph")

        print('Image CONCEPTS')

        train_img = train_data.img
        test_img = test_data.img
        x_tab = torch.cat((train_img, test_img))
        print_image(None, concepts_img_local.detach().numpy(), x_tab, y_img, 5, concepts_img_local.detach().numpy(), path, task='local', concepts=centroids)
        if MODE != 'single_CBM':
            print("\n_____________THIS IS FOR GRAPHS AND TABLE____________")
            concepts = torch.vstack([torch.Tensor(graph_concepts), torch.Tensor(tab_concepts)]).detach()

            y_double = torch.cat((y, y), dim=0)
            # find centroids for both modalities
            centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts, y_double)
            print(f"Number of graph cenroids: {len(centroids)}")
            persistence_utils.persist_experiment(centroids, path, 'centroids.z')
            persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels.z')
            persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels.z')

            # calculate cluster sizing
            cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
            train_mask_double = np.concatenate((train_mask, train_mask), axis=0)
            test_mask_double = np.concatenate((test_mask, test_mask), axis=0)
            concept_tree = (concepts.detach().numpy() > 0.5).astype(int)
            classifier = models.ActivationClassifierConcepts(y_double, concept_tree, train_mask_double, test_mask_double)

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            persistence_utils.persist_experiment(concept_metrics, path, 'shared_concept_metrics.z')
            wandb.log({'shared completeness': classifier.accuracy, 'num clusters shared': len(centroids)})

            # plot concept heatmaps
            # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

            # plot clustering
            tab_or_graph = torch.cat((torch.ones(int(concepts.shape[0]/2)), torch.zeros(int(concepts.shape[0]/2))), dim=0)
            visualisation_utils.plot_clustering(seed, concepts, tab_or_graph, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="shared", id_path="_graph")

            edges_train = train_data.edge_index.transpose(0, 1).detach().cpu().numpy()
            edges_test = test_data.edge_index.transpose(0, 1).detach().cpu().numpy() + train_data.x.shape[0]

            edges_t = np.concatenate((edges_train, edges_test), axis=0)
            expanded_batch = np.concatenate((train_data.batch.cpu().numpy(), test_data.batch.cpu().numpy() + train_data.batch[-1].item() + 1), axis=0)
            g_concepts = graph_concepts.detach().cpu().numpy()
            print('GRAPH CONCEPTS')
            top_plot, top_concepts = plot_samples(None, g_concepts, expanded_x, expanded_pos, expanded_batch, y,
                                                  len(centroids), 5, edges_t, concepts, path, task='global',
                                                  concepts=centroids)

            print('Image CONCEPTS')
            t_concepts = tab_concepts.detach().numpy()
            train_img = train_data.img
            test_img = test_data.img
            x_tab = torch.cat((train_img, test_img))
            top_plot_images, top_concepts_img = print_image(None, t_concepts, x_tab, y, 5, concepts, path,
                                                            task='global', concepts=centroids)



            print_near_example(t_concepts, x_tab, g_concepts, expanded_x, expanded_pos, expanded_edges, expanded_batch, y_img, y_g, path)

            print('------SHARED SPACE-----')
            top_concepts_both = np.array(top_concepts + top_concepts_img)

            top_plot_both = top_plot + top_plot_images

            if len(top_concepts + top_concepts_img) > 0:
                visualisation_utils.plot_clustering_images_inside(seed, concepts, top_concepts_both, top_plot_both,
                                                                  used_centroid_labels, path,
                                                                  'shared space with images')

            print('----------MISSING MODALITY------------')
            missing_accuracy = test_missing_modality(full_test_loader, model, g_concepts, t_concepts)
            wandb.log({'missing graph modality accuracy': missing_accuracy[0],
                       'missing image modality accuracy': missing_accuracy[1]})

            print("\n_____________THIS IS FOR COMBINED CONCEPTS____________")
            union_concepts = torch.cat([torch.Tensor(graph_concepts), torch.Tensor(tab_concepts)], dim=-1).detach()

            # find centroids for both modalities
            centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(union_concepts, y)
            print(f"Number of graph cenroids: {len(centroids)}")
            persistence_utils.persist_experiment(centroids, path, 'union_centroids.z')
            persistence_utils.persist_experiment(centroid_labels, path, 'union_centroid_labels.z')
            persistence_utils.persist_experiment(used_centroid_labels, path, 'union_used_centroid_labels.z')

            # calculate cluster sizing
            cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)

            union_concepts_tree = (union_concepts.detach().numpy() > 0.5).astype(int)
            classifier = models.ActivationClassifierConcepts(y, union_concepts_tree, train_mask, test_mask)

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
            wandb.log({'combined completeness': classifier.accuracy, 'num clusters combined': len(centroids)})

            acc_noise1, acc_noise2 = test_with_incremental_noise(model, full_test_loader, if_interpretable_model=interpretable, mode=MODE)
            wandb.log({'noise accuracy mod1': acc_noise1, 'noise accuracy mod2': acc_noise2})
            print(f'Noise accuracy mod1: {acc_noise1}, mod2: {acc_noise2}')

            acc_int_gt1, acc_int_mod1, acc_int_gt2, acc_int_mod2 = test_with_interventions(model, t_concepts.shape[-1], test_loader, g_concepts, t_concepts, if_interpretable_model=interpretable, mode=MODE)
            wandb.log({'interventions accuracy latent mod1': acc_int_gt1, 'interventions accuracy missing modality1': acc_int_mod1, 
                       'interventions accuracy latent mod2': acc_int_gt2, 'interventions accuracy missing modality2': acc_int_mod2})
            print(f'Interventions accuracy mod 1: {acc_int_gt1}, {acc_int_mod1}, mod 2: {acc_int_gt2}, {acc_int_mod2}')

            try:
                classifier.plot2(path, [union_concepts.detach(), y, expanded_batch, edges_t, x_tab, path], mode='mnist', x=expanded_x, pos=expanded_pos)
                classifier.plot2(path, [union_concepts.detach(), y, expanded_batch, edges_t, x_tab, path], integer=5, layers=[0, 3], mode='mnist', x=expanded_x, pos=expanded_pos)
            except Exception as e:
                print(e)

            classifier = models.ActivationClassifierConcepts(y, union_concepts_tree, train_mask, test_mask,
                                                             max_depth=5)

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
            wandb.log({'combined completeness 2': classifier.accuracy})

            try:
                classifier.plot2(path, [union_concepts.detach(), y, expanded_batch, edges_t, x_tab, path], integer=3, mode='mnist', x=expanded_x, pos=expanded_pos)
            except Exception as e:
                print(e)
            # plot concept heatmaps
            # visualisation_utils.plot_concept_heatmap(centroids, union_concepts, y, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

            # plot clustering
            visualisation_utils.plot_clustering(seed, union_concepts, y, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="combined", id_path="_graph",
                                                extra=True, train_mask=train_mask, test_mask=test_mask, n_classes=NUM_CLASSES)

    # clean up
    plt.close()

    return model_to_return

if __name__ == '__main__':
    main()