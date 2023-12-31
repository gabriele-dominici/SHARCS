import os

import wandb
import yaml
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import torch.nn.functional as F
from torch_geometric.data import DataLoader

import seaborn as sns

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics.cluster import homogeneity_score, completeness_score

import clustering_utils
import data_utils
import model_utils
import persistence_utils
import visualisation_utils
import models
import networkx as nx
from tqdm import tqdm

import random

visualisation_utils.set_rc_params()

def test_graph_class(model, dataloader, if_interpretable_model=True, mode='sharcs'):
    # enter evaluation mode
    correct = 0
    correct_t = 0
    device = torch.device(dev)
    for data in dataloader:
        data.to(device)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean)
            else:
                _, _, out_g, out_t = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                           data.graph_stats_clean)
                pred = out_g.argmax(dim=1)
                correct += int((pred == data.y).sum())
                pred = out_t.argmax(dim=1)
                correct_t += int((pred == data.y).sum())
                continue
        elif mode == 'VoteClassifier': 
            _, _, out_g, out_t = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                           data.graph_stats_clean)
            val_1 = out_g.max(dim=1)
            pred_1 = out_g.argmax(dim=1)
            val_2 = out_t.max(dim=1)
            pred_2 = out_t.argmax(dim=1)
            result = val_1.values > val_2.values
            a_filt = pred_1[result]
            b_filt = pred_2[~result]
            pred = torch.cat([a_filt, b_filt], dim=0)
            l_1 = data.y[result]
            l_2 = data.y[~result]
            labels = torch.cat([l_1, l_2], dim=0)
            correct += int((pred == data.y).sum())
            continue
        else:
            if mode in ['vanilla', 'late_fus_sum']:
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                           data.graph_stats_clean)
            elif mode == 'single_vanilla' or mode == 'anchors':
                _, _, out_g, out_t = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                           data.graph_stats_clean)
                pred = out_g.argmax(dim=1)
                correct += int((pred == data.y).sum())
                pred = out_t.argmax(dim=1)
                correct_t += int((pred == data.y).sum())
                continue
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())

    if mode == 'single_vanilla' or mode == 'single_CBM' or mode == 'anchors':
        return correct / len(dataloader.dataset), correct_t / len(dataloader.dataset)

    return correct / len(dataloader.dataset)

def train_graph_class(model, train_loader, test_loader, epochs, lr, if_interpretable_model=True, mode='sharcs'):
    # register hooks to track activation
    model = model_utils.register_hooks(model)
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
        running_dist = 0
        num_batches = 0
        for data in train_loader:
            data.to(device)
            model.train()

            optimizer.zero_grad()

            if if_interpretable_model:
                if mode == 'SHARCS':
                    concepts_gnn, concepts_tab, out, d = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)
                    loss = criterion(out, one_hot) + 0.1 * d
                elif mode == 'single_CBM':
                    concepts_gnn, concepts_tab, out_g, out_t = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                               data.graph_stats_clean)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out_g)
                    loss = criterion(out_g, one_hot) + criterion(out_t, one_hot)
                    d = 0
                else:
                    concepts_gnn, concepts_tab, out, d = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                               data.graph_stats_clean)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)
                    loss = criterion(out, one_hot)
            else:
                if mode in ['vanilla', 'late_fus_sum']:
                    concepts_gnn, concepts_tab, out, d = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                               data.graph_stats_clean)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)
                    loss = criterion(out, one_hot)
                elif mode == 'single_vanilla' or mode == 'anchors':
                    concepts_gnn, concepts_tab, out_g, out_t = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                               data.graph_stats_clean)
                    one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out_g)
                    loss = criterion(out_g, one_hot) + criterion(out_t, one_hot)
                    d = 0

            # calculate loss


            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dist += d
            num_batches += 1

            optimizer.step()

        # get accuracy
        if mode == 'single_vanilla' or mode == 'single_CBM' or mode == 'anchors':
            train_acc_g, train_acc_t = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            test_acc_g, test_acc_t = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            train_acc = train_acc_g
            if mode == 'single_vanilla':
                test_acc = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model, mode='VoteClassifier')
            else:
                test_acc = test_acc_g
        else:
            train_acc = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            test_acc = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model, mode=mode)

        # add to list and print
        model.eval()
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # get testing loss
        test_running_loss = 0
        test_num_batches = 0
        for data in test_loader:
            data.to(device)
            if if_interpretable_model:
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean)
            else:
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                                           data.graph_stats_clean)

            one_hot = torch.nn.functional.one_hot(data.y, num_classes=2).type_as(out)

            test_running_loss += criterion(out, one_hot).item()
            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        train_d.append(running_dist / num_batches)
        test_loss.append(test_running_loss / test_num_batches)


        if mode in ['single_CBM', 'anchors']:
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Graph Acc: {:.5f}, Tab Acc: {:.5f}, Dist:{:.5f}'.
                format(epoch, train_loss[-1], test_loss[-1], test_acc_g, test_acc_t, train_d[-1]), end="\r")

            wandb.log({'Epoch': epoch, 'Train loss': train_loss[-1], 'Test Loss': test_loss[-1],
                       'Graph Acc': test_acc_g, 'Tab Acc': test_acc_t, 'Dist': train_d[-1]})
        else:
            print(
                'Epoch: {:03d}, Train Loss: {:.5f}, Test Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}, Dist:{:.5f}'.
                format(epoch, train_loss[-1], test_loss[-1], train_acc, test_acc, train_d[-1]), end="\r")

            wandb.log({'Epoch': epoch, 'Train loss': train_loss[-1], 'Test Loss': test_loss[-1],
                       'Train Acc': train_acc, 'Test Acc': test_acc, 'Dist': train_d[-1]})


    return train_accuracies, test_accuracies, train_loss, test_loss

def print_samples(clustering_model, data, x, y, k, num_nodes_view, all_concepts, concepts=None, task='shared'):
    res_sorted = clustering_utils.get_node_distances(clustering_model, data, concepts)
    sample_graphs = []
    sample_feat = []

    unique_c_unfiltered, counts = np.unique((all_concepts>0.5) + 0, axis=0, return_counts=True)
    counts_filter = counts >= 5
    unique_c = unique_c_unfiltered[counts_filter]
    k = len(unique_c)
    res_sorted = res_sorted[:, counts_filter]

    l = []

    unique_g_c, counts = np.unique((data > 0.5) + 0, axis=0, return_counts=True)
    min_treshold = min(5, len(counts))
    threshold = np.sort(counts)[-min_treshold]
    counts_filter = counts >= threshold
    unique_g_c = unique_g_c[counts_filter].tolist()
    unique_g_c = [''.join(str(e) for e in item) for item in unique_g_c]

    for index, el in enumerate(unique_c):
        tmp = ''.join(str(e) for e in el)
        if tmp in unique_g_c:
            l += [index]

    if k > 10:
        k = 10
        l = l[:10]
    elif k == 0:
      return sample_graphs, sample_feat

    x = x.numpy()
    used_concepts = set()
    top_concepts = []
    top_plot = []
    for i in l:
        distances = res_sorted[:, i]
        if num_nodes_view < 0:
            top_indices = np.argsort(distances)[::][num_nodes_view:]
        else:
            top_indices = np.argsort(distances)[::][:num_nodes_view]

        raw_concepts = data[top_indices]
        top_instance = x[top_indices]

        labels = y[top_indices]
        columns=[]
        for digit in range(x.shape[-1]):
            columns.append("f_" + str(digit))
        columns += ['labels']
        table = wandb.Table(columns=columns)

        c = np.array2string((data[int(top_indices[0])]>0.5) + 0, precision=2, separator=',',
                      suppress_small=True)

        if c not in used_concepts:
            used_concepts.add(c)
            print(f'Concept: {i} ({c})')
            counter = 0
            for instance, label, txt_raw in zip(top_instance, labels, raw_concepts):
                if counter == 0:
                    top_concepts += [txt_raw]
                    fig_aux = plt.figure(figsize=(3, 3))
                    txt = str(int(instance[1].item())) + ' ' + str(int(instance[4].item()))
                    plt.text(0, 0.5, f'{txt}', fontsize=60)
                    plt.axis('off')
                    fig_aux.savefig(f"./images/{i}_txt.png")
                    im = matplotlib.image.imread(f"./images/{i}_txt.png")
                    top_plot += [im]
                table.add_data(*instance, label)
                print(f'Initial features: {instance}, Label: {label}')
                counter += 1
        wandb.log({f"table concepts {task} concepts {i}": table})
    return top_plot, top_concepts

def save_centroids(centroids, y, used_centroid_labels, union_concepts,
                           g_concepts, batch, edges,
                           t_concepts, x_tab,
                           path):

    g_c = centroids[:, :int(centroids.shape[-1] / 2)]
    t_c = centroids[:, int(centroids.shape[-1] / 2):]
    res_sorted_g = clustering_utils.get_node_distances(None, g_concepts, g_c)
    res_sorted_t = clustering_utils.get_node_distances(None, t_concepts, t_c)
    unique_concepts = np.unique(used_centroid_labels)
    g_con = (g_concepts > 0.5) + 0
    for c in tqdm(unique_concepts):
        distances_g = res_sorted_g[:, c]
        distances_t = res_sorted_t[:, c]
        top_indices_g = [np.argsort(distances_g)[::][0]]
        top_indices_t = np.argsort(distances_t)[::][0]

        top_instance = x_tab[top_indices_t]
        tab = str(int(top_instance[1].item())) + ' ' + str(int(top_instance[4].item()))

        tg, cm, labels, concepts_list = clustering_utils.get_top_graphs(top_indices_g, g_con, y, edges, batch)
        plt.figure(figsize=(3, 3))
        plt.title(tab, fontsize=10)
        nx.draw(tg[0])
        plt.savefig(f'{path}/{c}.svg')
        plt.close()
    plt.close('all')



def plot_samples(clustering_model, data, batch, y, k, num_nodes_view, edges, all_concepts, path, concepts=None, task='shared'):

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

    unique_g_c, counts = np.unique((data > 0.5) + 0, axis=0, return_counts=True)
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

    if k > 10:
        k = 10
        l = l[:10]
    elif k == 0:
      return sample_graphs, sample_feat

    fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
    fig.suptitle(f'Nearest Instances to Cluster Centroid for {task} Concepts', y=1.005)

    top_concepts = []
    top_plot = []
    for i, ax_list in zip(l, axes):
        distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            raw_concepts = data[top_indices]
            tg, cm, labels, concepts_list = clustering_utils.get_top_graphs(top_indices, c, y, edges, batch)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if k == 1:
            ax_list = [ax_list]
        counter = 0
        for ax, new_G, color_map, g_label, g_concept, g_raw in zip(ax_list, top_graphs, color_maps, labels, concepts_list,
                                                                   raw_concepts):
            if counter == 0:
                top_concepts += [g_raw]
                fig_aux = plt.figure(figsize=(4, 4))
                nx.draw(new_G, ax=fig_aux.add_subplot())
                plt.axis('off')
                fig_aux.savefig(f"./images/{i}_graph.png")
                im = matplotlib.image.imread(f"./images/{i}_graph.png")
                top_plot += [im]
            nx.draw(new_G, node_color=color_map, with_labels=True, ax=ax)
            ax.set_title(f"label {g_label}, concept {i}", fontsize=14)
            counter += 1
        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    plt.savefig(os.path.join(path, f"{task}_g_concepts.pdf"))
    plt.savefig(os.path.join(path, f"{task}_g_concepts.png"))
    wandb.log({task: wandb.Image(plt)})
    plt.show()
    plt.close('all')

    return top_plot, top_concepts

def print_near_example(o_concepts, x_o, g_concepts, x_g, edges, batch, y, path, times=2, example=4):
    d = nn.PairwiseDistance(p=2)

    figure = plt.figure(figsize=(18, 3 * times + 2))
    image_indexes = []
    graph_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(o_concepts), size=(1,)).item()
        dist = d(torch.Tensor(g_concepts), torch.Tensor(o_concepts[sample_idx])).squeeze(-1)
        g_index = torch.argsort(dist)[:example]
        image_indexes += [sample_idx]
        graph_indexes += [list(g_index.numpy())]
    cols, rows = 5, times
    edges = torch.transpose(edges, 1, 0).cpu()
    df = pd.DataFrame(edges.numpy())
    for i in range(1, times + 1):
        sample_idx = image_indexes[i - 1]
        txt = str(int(x_o[sample_idx][1].item())) + str(int(x_o[sample_idx][4].item()))
        figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 1)
        plt.axis("off")
        # plt.imshow(torch.tensor(img.squeeze().permute(1, 2, 0), dtype=torch.uint8))
        plt.text(0, 0.5, f'{txt}', fontsize=50)
        for j, idx in enumerate(graph_indexes[i - 1]):
            node_indexes = torch.Tensor(list(range(batch.shape[0])))
            neighbours = node_indexes[batch == idx].numpy()
            neighbours = list(set(neighbours))
            df_neighbours = df[(df[0].isin(neighbours)) & (df[1].isin(neighbours))]
            edges = df_neighbours.to_numpy()
            new_G = nx.Graph()
            new_G.add_edges_from(edges)
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            nx.draw(new_G)
            plt.axis('off')
    plt.savefig(os.path.join(path, f"similar_tab.pdf"))
    plt.savefig(os.path.join(path, f"similar_tab.png"))
    wandb.log({'similar_tab': wandb.Image(plt)})
    plt.show()
    plt.close()
    plt.close('all')

    figure = plt.figure(figsize=(18, 3 * times + 2))
    image_indexes = []
    graph_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(o_concepts), size=(1,)).item()
        dist = d(torch.Tensor(o_concepts), torch.Tensor(g_concepts[sample_idx])).squeeze(-1)
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
        new_G = nx.Graph()
        new_G.add_edges_from(edges)
        figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 1)
        nx.draw(new_G)
        plt.axis('off')
        for j, idx in enumerate(image_indexes[i - 1]):
            txt = str(int(x_o[idx][1].item())) + str(int(x_o[idx][4].item()))
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            plt.axis("off")
            plt.text(0, 0.5, f'{txt}', fontsize=50)

    plt.savefig(os.path.join(path, f"similar_g.pdf"))
    plt.savefig(os.path.join(path, f"similar_g.png"))
    wandb.log({'similar_g': wandb.Image(plt)})
    plt.show()
    plt.close()
    plt.close('all')

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
    print('----MISSING MOD 1----')
    for data in dataloader:
        data.to(device)
        tab, tab_aux = model(data.graph_aux, data.edge_index, data.batch, data.graph_stats,
                             data.graph_stats_clean, missing=True, mod1=True)
        retreived_graph = retreived_similar(tab_aux, concepts_mod1)
        _, _, out, _ = model(retreived_graph, data.edge_index, data.batch, tab,
                             data.graph_stats_clean, missing=True, prediction=True)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for data in dataloader:
        data.to(device)
        graph, graph_aux = model(data.x, data.edge_index, data.x_batch, data.x_tab,
                             data.graph_stats_clean, missing=True, mod2=True, aux_edge=data.aux_edge_index, batch_aux=data.x_tab_batch)
        retreived_tab = retreived_similar(graph_aux, concepts_mod2)
        _, _, out, _ = model(graph, data.edge_index, data.batch, retreived_tab,
                             data.graph_stats_clean, missing=True, prediction=True)
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
        tab, tab_aux = model(data.graph_aux, data.edge_index, data.batch, data.graph_stats,
                             data.graph_stats_clean, missing=True, mod1=True)
        tab_aux = compute_relative_rep(tab_aux, anchors_mod2)
        tab = compute_relative_rep(tab, anchors_mod2)
        retreived_graph = retreived_similar(tab_aux, concepts_mod1)
        out = pred_model(tab, retreived_graph)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for data in dataloader:
        data.to(device)
        graph, graph_aux = model(data.x, data.edge_index, data.x_batch, data.x_tab,
                             data.graph_stats_clean, missing=True, mod2=True, aux_edge=data.aux_edge_index, batch_aux=data.x_tab_batch)
        graph_aux = compute_relative_rep(graph_aux, anchors_mod1)
        graph = compute_relative_rep(graph, anchors_mod1)
        retreived_tab = retreived_similar(graph_aux, concepts_mod2)
        out = pred_model(retreived_tab, graph)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    acc2 = correct / len(dataloader.dataset)
    return acc1, acc2


def collect_hidden_representation(model, dataloader):
    model.eval()
    device = torch.device(dev)
    tab = []
    graph = []
    y = []
    for data in dataloader:
        data.to(device)
        tmp_graph, tmp_tab, _, _ = model(data.x, data.edge_index, data.batch, data.graph_stats,
                                   data.graph_stats_clean)
        tab += [tmp_tab.detach().cpu()]
        graph += [tmp_graph.detach().cpu()]
        y += [data.y.detach().cpu()]
    tab = torch.vstack(tab)
    graph = torch.vstack(graph)
    y = torch.hstack(y)
    return tab, graph, y

def choose_anchors(model, dataloader):
    data = next(iter(dataloader))
    print(data.graph_aux.shape)
    device = torch.device(dev)
    data.to(device)
    tmp_graph, tmp_tab, _, _ = model(data.x, data.edge_index, data.batch, data.graph_aux,
                                     data.graph_stats_clean)
    return tmp_tab.detach().cpu(), tmp_graph.detach().cpu()

def compute_relative_rep(mod, anchors):
        result = torch.Tensor()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        for t in anchors:
            tmp = cos(mod, t).unsqueeze(dim=-1)
            result = torch.cat((result, tmp), dim=-1)
        return result

def awgn(signal, desired_snr, signal_power):
    """
    Add AWGN to the input signal to achieve the desired SNR level.
    """
    # Calculate the noise power based on the desired SNR and signal power
    noise_power = signal_power / (10**(desired_snr / 10))
    
    # Generate the noise with the calculated power
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    # Add the noise to the original signal
    noisy_signal = signal + torch.Tensor(noise)
    
    return noisy_signal

def test_with_incremental_noise(model, dataloader, if_interpretable_model=True, mode='sharcs'):
    correct = 0
    correct_t = 0
    device = torch.device(dev)
    for data in dataloader:
        data.to(device)
        data.graph_stats = awgn(data.graph_stats, 5, 1)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean, noise="mod1")
                concepts_gnn, concepts_tab, out2, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean, noise="mod2")
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
                concepts_gnn_noise, concepts_tab_noise, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean, noise="mod1")
                concepts_gnn_noise2, concepts_tab_noise2, out2, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean, noise="mod2")
                concepts_gnn, concepts_tab, out, _ = model(data.x, data.edge_index, data.batch, data.graph_stats, data.graph_stats_clean)
                concepts_noise = torch.cat([concepts_gnn_noise, concepts_tab_noise], dim=-1)
                concepts_noise2 = torch.cat([concepts_gnn_noise2, concepts_tab_noise2], dim=-1)
                concepts = torch.cat([concepts_gnn, concepts_tab], dim=-1)
                
                tab, tab_aux = model(data.graph_aux, data.edge_index, data.batch, data.graph_stats,
                                    data.graph_stats_clean, missing=True, mod1=True)
                retreived_graph = retreived_similar(tab_aux, concepts_mod1)
                concepts_retrieved1 = torch.cat([retreived_graph, tab], dim=-1)
                concepts_noise_missing_mod1 = concepts_noise.clone()
                
                graph, graph_aux = model(data.x, data.edge_index, data.x_batch, data.x_tab,
                             data.graph_stats_clean, missing=True, mod2=True, aux_edge=data.aux_edge_index, batch_aux=data.x_tab_batch)
                retreived_tab = retreived_similar(graph_aux, concepts_mod2)
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
                concepts_gnn, concepts_tab, out, _ = model(g_concepts, data.edge_index, data.batch, t_concepts, data.graph_stats_clean, missing=True, prediction=True)
                g_concepts = concepts_noise2[:, :int(concepts_noise2.shape[1] / 2)]
                t_concepts = concepts_noise2[:, int(concepts_noise2.shape[1] / 2):]
                concepts_gnn, concepts_tab, out2, _ = model(g_concepts, data.edge_index, data.batch, t_concepts, data.graph_stats_clean, missing=True, prediction=True)
                
                g_concepts_missing_mod = concepts_noise_missing_mod1[:, :int(concepts_noise_missing_mod1.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod1[:, int(concepts_noise_missing_mod1.shape[1] / 2):]
                concepts_gnn_missing_mod, concepts_tab_missing_mod, out_missing_mod, _ = model(g_concepts_missing_mod, data.edge_index, data.batch, t_concepts_missing_mod, data.graph_stats_clean, missing=True, prediction=True)
                g_concepts_missing_mod = concepts_noise_missing_mod2[:, :int(concepts_noise_missing_mod2.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod2[:, int(concepts_noise_missing_mod2.shape[1] / 2):]
                concepts_gnn_missing_mod, concepts_tab_missing_mod, out_missing_mod2, _ = model(g_concepts_missing_mod, data.edge_index, data.batch, t_concepts_missing_mod, data.graph_stats_clean, missing=True, prediction=True)
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
    tag = 'xor'
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
                       'size': SIZE,
                       'model_name': MODEL_NAME,
                       'num_classes': NUM_CLASSES,
                       'train_test_split': TRAIN_TEST_SPLIT,
                       'batch_size': BATCH_SIZE,
                       'num_hidden_units': NUM_HIDDEN_UNITS,
                       'cluster_encoding_size': CLUSTER_ENCODING_SIZE,
                       'epochs': EPOCHS,
                       'lr': LR,
                       'layer_num': LAYER_NUM
                      }

    persistence_utils.persist_experiment(config, path, 'config.z')

    # load data
    device = torch.device(dev)

    dataset = data_utils.create_xor_dataset(SIZE)

    train_idx = int(len(dataset) * TRAIN_TEST_SPLIT)
    train_set = dataset[:train_idx]
    test_set = dataset[train_idx:]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, follow_batch=['x', 'x_tab'])
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, follow_batch=['x', 'x_tab'])

    full_train_loader = DataLoader(train_set, batch_size=int(len(train_set) * 0.1), shuffle=True, follow_batch=['x', 'x_tab'])
    full_test_loader = DataLoader(test_set, batch_size=int(len(test_set) * 0.1), follow_batch=['x', 'x_tab'])


    # model training
    if MODE == 'vanilla':
        model = models.Tab_Graph_Vanilla(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS,
                                 NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = False
    elif MODE == 'late_fus_sum':
        model = models.Tab_Graph_Vanilla_LateFusion(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS,
                                 NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_vanilla' or MODE == 'anchors':
        model = models.Tab_Graph_Vanilla_single(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS,
                                 NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_CBM':
        model = models.Tab_Graph_SingleCBM(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS,
                                 NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True
    elif MODE == 'multi_CBM':
        model = models.Tab_Graph_MultiCBM(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS,
                                 NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True
    else:
        model = models.Tab_Graph_missing(train_set[0].num_node_features, train_set[0].graph_stats.shape[-1], NUM_HIDDEN_UNITS, NUM_HIDDEN_UNITS, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True

    model_to_return = model
    model.to(device)
    # register hooks to track activation

    model = model_utils.register_hooks(model)

    # train
    train_acc, test_acc, train_loss, test_loss = train_graph_class(model, train_loader, test_loader, EPOCHS, LR,
                                                                   if_interpretable_model=interpretable, mode=MODE)
    persistence_utils.persist_model(model, path, 'model.z')

    visualisation_utils.plot_model_accuracy(train_acc, test_acc, MODEL_NAME, path)
    visualisation_utils.plot_model_loss(train_loss, test_loss, MODEL_NAME, path)

    if MODE == 'anchors':
        train_tab_representation, train_graph_representation, train_y = collect_hidden_representation(model, train_loader)
        test_tab_representation, test_graph_representation, test_y = collect_hidden_representation(model, test_loader)

        train_loader = DataLoader(train_set, batch_size=int(len(train_set) * 0.1), shuffle=True, follow_batch=['x', 'x_tab'])
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

        pred_model = models.PredModel(input_size, input_size*3, NUM_CLASSES)

        model_utils.train_prediction(pred_model, train_loader, test_loader, 100, LR, NUM_CLASSES, dev)

        print("\n_____________THIS IS FOR GRAPHS AND TABLE____________")

        train_data = next(iter(full_train_loader)).to(device)
        train_graph, train_tab, _, _ = model(train_data.x, train_data.edge_index, train_data.batch,
                                             train_data.graph_stats, train_data.graph_stats_clean)
        train_graph = train_graph.cpu()
        train_tab = train_tab.cpu()

        train_tab = compute_relative_rep(train_tab, anchors_tab)
        train_graph = compute_relative_rep(train_graph, anchors_graph)


        test_data = next(iter(full_test_loader)).to(device)
        test_graph, test_tab, _, _ = model(test_data.x, test_data.edge_index, test_data.batch, test_data.graph_stats,
                                            test_data.graph_stats_clean)

        test_graph = test_graph.cpu()
        test_tab = test_tab.cpu()

        test_tab = compute_relative_rep(test_tab, anchors_tab)
        test_graph = compute_relative_rep(test_graph, anchors_graph)

        graph_concepts = torch.vstack([train_graph, test_graph])
        tab_concepts = torch.vstack([train_tab, test_tab])

        edges_train = train_data.edge_index.cpu()
        edges_test = test_data.edge_index.cpu()
        offset = train_data.x.shape[0]
        edges_test = edges_test + offset
        expanded_edges = torch.cat((edges_train, edges_test), dim=-1)

        x_train = train_data.x.cpu()
        x_test = test_data.x.cpu()
        expanded_x = torch.cat((x_train, x_test))

        y_train = train_data.y.cpu()
        y_test = test_data.y.cpu()
        y = torch.cat((y_train, y_test))

        train_mask = np.zeros(y.shape[0], dtype=bool)
        train_mask[:train_data.y.shape[0]] = True
        test_mask = ~train_mask

        offset = train_data.batch[-1] + 1
        batch = torch.cat((train_data.batch, test_data.batch + offset))

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

        edges_train = train_data.edge_index.transpose(0, 1).detach().numpy()
        edges_test = test_data.edge_index.transpose(0, 1).detach().numpy() + train_data.x.shape[0]

        edges_t = np.concatenate((edges_train, edges_test), axis=0)
        expanded_batch = np.concatenate(
            (train_data.batch.numpy(), test_data.batch.numpy() + train_data.batch[-1].item() + 1), axis=0)
        g_concepts = graph_concepts.detach().numpy()

        t_concepts = tab_concepts.detach().numpy()
        x_tab = torch.cat((train_data.graph_stats, test_data.graph_stats))
        x_tab = x_tab.reshape(-1, 6)

        print('-------- SIMILAR EXAMPLE FROM DIFFERENT MODALITIES----------')
        print_near_example(t_concepts, x_tab, g_concepts, expanded_x, expanded_edges, expanded_batch, y, path)
        print('----------MISSING MODALITY------------')
        missing_accuracy = test_missing_modality_anchors(full_test_loader, model, pred_model,
                                                         g_concepts, t_concepts,
                                                         anchors_graph, anchors_tab)
        wandb.log({'missing graph modality accuracy': missing_accuracy[0],
                   'missing tab modality accuracy': missing_accuracy[1]})
        print(f'Missing accuracy: {missing_accuracy}')

    if interpretable:
        # get model activations for complete dataset
        train_data = next(iter(full_train_loader)).to(device)
        train_node_concepts, _, _, _ = model(train_data.x, train_data.edge_index, train_data.batch, train_data.graph_stats, train_data.graph_stats_clean)
        train_clean_features = train_data.graph_stats_clean.cpu()
        train_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        train_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        train_tab_local_concepts = model.x_tab_local_concepts.cpu()
        train_tab_concepts = model.tab_shared_concepts.cpu()

        test_data = next(iter(full_test_loader)).to(device)
        test_node_concepts, _, _, _ = model(test_data.x, test_data.edge_index, test_data.batch, test_data.graph_stats, test_data.graph_stats_clean)
        test_clean_features = test_data.graph_stats_clean.cpu()
        test_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        test_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        test_tab_local_concepts = model.x_tab_local_concepts.cpu()
        test_tab_concepts = model.tab_shared_concepts.cpu()

        train_clean_features = train_clean_features.reshape(-1, 3)
        test_clean_features = test_clean_features.reshape(-1, 3)
        clean_features = torch.vstack((train_clean_features, test_clean_features))

        edges_train = train_data.edge_index.cpu()
        edges_test = test_data.edge_index.cpu()
        offset = train_data.x.shape[0]
        edges_test = edges_test + offset
        expanded_edges = torch.cat((edges_train, edges_test), dim=-1)

        graph_concepts = torch.vstack([train_graph_concepts, test_graph_concepts])
        graph_local_concepts = torch.vstack([train_graph_local_concepts, test_graph_local_concepts])
        tab_local_concepts = torch.vstack([train_tab_local_concepts, test_tab_local_concepts])
        tab_concepts = torch.vstack([train_tab_concepts, test_tab_concepts])

        x_train = train_data.x.cpu()
        x_test = test_data.x.cpu()
        expanded_x = torch.cat((x_train, x_test))

        y_train = train_data.y.cpu()
        y_test = test_data.y.cpu()
        y = torch.cat((y_train, y_test))

        train_mask = np.zeros(y.shape[0], dtype=bool)
        train_mask[:train_data.y.shape[0]] = True
        test_mask = ~train_mask

        offset = train_data.batch[-1] + 1
        batch = torch.cat((train_data.batch, test_data.batch + offset))

        persistence_utils.persist_experiment(graph_concepts, path, 'graph_concepts.z')
        persistence_utils.persist_experiment(tab_concepts, path, 'tab_concepts.z')

        print("\n_____________THIS IS FOR GRAPHS____________")
        concepts_g_local = torch.Tensor(graph_local_concepts).detach()
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts_g_local, y)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids_g.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels_g.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels_g.z')

        # calculate cluster sizing
        cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
        concepts_g_local_tree = (concepts_g_local.detach().numpy() > 0.5).astype(int)
        classifier = models.ActivationClassifierConcepts(y, concepts_g_local_tree, train_mask, test_mask)

        print(f"Classifier Concept completeness score: {classifier.accuracy}")
        concept_metrics = [('cluster_count', cluster_counts)]
        persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
        wandb.log({'local graph completeness': classifier.accuracy, 'num clusters local graph': len(centroids)})

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_g_local, y, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="graph local", id_path="_graph")

        edges_train = train_data.edge_index.transpose(0, 1).detach().cpu().numpy()
        edges_test = test_data.edge_index.transpose(0, 1).detach().cpu().numpy() + train_data.x.shape[0]

        edges_t = np.concatenate((edges_train, edges_test), axis=0)
        expanded_batch = np.concatenate((train_data.batch.cpu().numpy(), test_data.batch.cpu().numpy() + train_data.batch[-1].item() + 1), axis=0)
        g_concepts = concepts_g_local.detach().cpu().numpy()
        print('GRAPH CONCEPTS')
        sample_graphs, sample_feat = plot_samples(None, g_concepts, expanded_batch, y, len(centroids), 5, edges_t, concepts_g_local, path, concepts=centroids, task='local')

        print("\n_____________THIS IS FOR TABLE____________")
        concepts_tab_local = torch.Tensor(tab_local_concepts).detach()
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts_tab_local, y)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids_img.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels_img.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels_img.z')

        # calculate cluster sizing
        cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
        concepts_tab_local_tree = (concepts_tab_local.detach().numpy() > 0.5).astype(int)
        classifier = models.ActivationClassifierConcepts(y, concepts_tab_local_tree, train_mask, test_mask)

        print(f"Classifier Concept completeness score: {classifier.accuracy}")
        concept_metrics = [('cluster_count', cluster_counts)]
        wandb.log({'local table completeness': classifier.accuracy, 'num clusters local table': len(centroids)})
        persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_tab_local, y, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="table local", id_path="_graph")

        print('Table CONCEPTS')
        t_concepts = tab_local_concepts.detach().numpy()
        x_tab = torch.cat((train_data.graph_stats, test_data.graph_stats))
        x_tab = x_tab.reshape(-1, 6)
        print_samples(None, t_concepts, x_tab, y, len(centroids), 5, concepts_tab_local, concepts=centroids, task='local')

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
            concepts_tree = (concepts.detach().numpy() > 0.5).astype(int)
            classifier = models.ActivationClassifierConcepts(y_double, concepts_tree, train_mask_double, test_mask_double)
            # concept alignment

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            wandb.log({'shared completeness': classifier.accuracy, 'num clusters shared': len(centroids)})
            persistence_utils.persist_experiment(concept_metrics, path, 'shared_concept_metrics.z')

            # plot concept heatmaps
            # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

            # plot clustering
            tab_or_graph = torch.cat((torch.ones(int(concepts.shape[0]/2)), torch.zeros(int(concepts.shape[0]/2))), dim=0)
            visualisation_utils.plot_clustering(seed, concepts, tab_or_graph, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="shared", id_path="_graph")

            edges_train = train_data.edge_index.transpose(0, 1).detach().numpy()
            edges_test = test_data.edge_index.transpose(0, 1).detach().numpy() + train_data.x.shape[0]

            edges_t = np.concatenate((edges_train, edges_test), axis=0)
            expanded_batch = np.concatenate((train_data.batch.numpy(), test_data.batch.numpy() + train_data.batch[-1].item() + 1), axis=0)
            g_concepts = graph_concepts.detach().numpy()
            print('GRAPH CONCEPTS')
            top_plot, top_concepts = plot_samples(None, g_concepts, expanded_batch, y, len(centroids), 5, edges_t, concepts, path, concepts=centroids, task='shared')

            print('TABLE CONCEPTS')
            t_concepts = tab_concepts.detach().numpy()
            x_tab = torch.cat((train_data.graph_stats, test_data.graph_stats))
            x_tab = x_tab.reshape(-1, 6)
            top_plot_images, top_concepts_img = print_samples(None, t_concepts, x_tab, y, len(centroids), 5, concepts, concepts=centroids, task='shared')

            print('------SHARED SPACE-----')
            top_concepts_both = np.array(top_concepts + top_concepts_img)

            top_plot_both = top_plot + top_plot_images

            if len(top_concepts + top_concepts_img) > 0:
                visualisation_utils.plot_clustering_images_inside(seed, concepts, top_concepts_both, top_plot_both, used_centroid_labels, path, 'shared space with images')
            print('-------- SIMILAR EXAMPLE FROM DIFFERENT MODALITIES----------')
            print_near_example(t_concepts, x_tab, g_concepts, expanded_x, expanded_edges, expanded_batch, y, path)
            print('----------MISSING MODALITY------------')
            missing_accuracy = test_missing_modality(full_test_loader, model, g_concepts, t_concepts)
            wandb.log({'missing graph modality accuracy': missing_accuracy[0], 'missing tab modality accuracy': missing_accuracy[1]})
            print(f'Missing accuracy: {missing_accuracy}')
            print("\n_____________THIS IS FOR COMBINED CONCEPTS____________")
            union_concepts = torch.cat([torch.Tensor(graph_concepts), torch.Tensor(tab_concepts)], dim=-1)
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

            # save_centroids(centroids, y, used_centroid_labels, union_concepts,
            #                g_concepts, expanded_batch, edges_t,
            #                t_concepts, x_tab,
            #                path)
            classifier.plot2(path, [union_concepts.detach(), y, expanded_batch, edges_t, x_tab, path])

            # concept alignment

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            wandb.log({'combined completeness': classifier.accuracy, 'num clusters combined': len(centroids)})
            persistence_utils.persist_experiment(concept_metrics, path, 'combined_concept_metrics.z')

            # plot concept heatmaps
            # visualisation_utils.plot_concept_heatmap(centroids, union_concepts, y, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

            # plot clustering
            visualisation_utils.plot_clustering(seed, union_concepts.detach(), y, centroids, centroid_labels, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="combined", id_path="_graph",
                                                extra=True, train_mask=train_mask, test_mask=test_mask, n_classes=NUM_CLASSES)
            
            acc_noise1, acc_noise2 = test_with_incremental_noise(model, full_test_loader, if_interpretable_model=interpretable, mode=MODE)
            wandb.log({'noise accuracy mod1': acc_noise1, 'noise accuracy mod2': acc_noise2})
            print(f'Noise accuracy mod1: {acc_noise1}, mod2: {acc_noise2}')

            acc_int_gt1, acc_int_mod1, acc_int_gt2, acc_int_mod2 = test_with_interventions(model, t_concepts.shape[-1], test_loader, g_concepts, t_concepts, if_interpretable_model=interpretable, mode=MODE)
            wandb.log({'interventions accuracy latent mod1': acc_int_gt1, 'interventions accuracy missing modality1': acc_int_mod1, 
                       'interventions accuracy latent mod2': acc_int_gt2, 'interventions accuracy missing modality2': acc_int_mod2})
            print(f'Interventions accuracy mod 1: {acc_int_gt1}, {acc_int_mod1}, mod 2: {acc_int_gt2}, {acc_int_mod2}')


    # clean up
    plt.close('all')

    return model_to_return

if __name__ == '__main__':
    main()
