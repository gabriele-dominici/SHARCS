import os
import matplotlib.pyplot as plt
import matplotlib

import torch.nn as nn
from torch_geometric.data import Data, DataLoader


import torch

from torch_geometric.loader import DataLoader

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything

import clustering_utils
import data_utils
import model_utils
import persistence_utils
import visualisation_utils
import models
import wandb
import yaml

from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import numpy as np

from PIL import Image

from tqdm import tqdm

visualisation_utils.set_rc_params()

synonyms = {
  "thing": "thing",
  "object": "thing",
  "sphere": "sphere",
  "ball": "sphere",
  "cylinder": "cylinder",
  "cube": "cube",
  "block": "cube",
  "large": "large",
  "big": "large",
  "small": "small",
  "tiny": "small",
  "metal": "metal",
  "metallic": "metal",
  "shiny": "metal",
  "rubber": "rubber",
  "matte": "rubber",
  "red": "red",
  "purple": "purple",
  "cyan": "cyan",
  "green": "green",
  "blue": "blue",
  "gray": "gray",
  "brown": "brown",
  "yellow": "yellow",
}

def test_graph_class(model, dataloader, if_interpretable_model=True, mode='SHARCS'):
    # enter evaluation mode
    correct = 0
    correct_t = 0
    device = torch.device(dev)
    for input_image, input_text, _, _, y, _, _, _ in dataloader:
        # data.to(device)
        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        if mode in ['single_CBM', 'single_vanilla','anchors']:
            _, _, out_g, out_t = model(input_text, input_image, y)
            pred = out_g.argmax(dim=1)
            correct += int((pred == y).sum())
            pred = out_t.argmax(dim=1)
            correct_t += int((pred == y).sum())
        else:
            concepts_gnn, concepts_tab, out, _ = model(input_text, input_image, y)
            pred = out.argmax(dim=1)
            correct += int((pred == y).sum())
    if mode in ['single_CBM', 'single_vanilla', 'anchors']:
        return correct / len(dataloader.dataset), correct_t / len(dataloader.dataset)
    else:
        return correct / len(dataloader.dataset)

def train_graph_class(model, train_loader, test_loader, epochs, lr, if_interpretable_model=True, mode='SHARCS'):
    # register hooks to track activation
    model = model_utils.register_hooks(model)
    device = torch.device(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss().to(device)

    # list of accuracies
    train_accuracies, test_accuracies, train_loss, test_loss = list(), list(), list(), list()
    train_d = list()

    for epoch in tqdm(range(epochs)):
        model.train()

        running_loss = 0
        running_dist = 0
        num_batches = 0
        for input_image, input_text, _, _, y, _, _, _ in train_loader:
            input_image = input_image.to(device)
            input_text = input_text.to(device)
            y = y.to(device)
            model.train()

            optimizer.zero_grad()

            if mode == 'SHARCS':
                concepts_gnn, concepts_tab, out, d = model(input_text, input_image, y)
                one_hot = torch.nn.functional.one_hot(y, num_classes=2).type_as(out)
                if epoch < epochs-20:
                    loss = criterion(out, one_hot)
                else:
                    loss = criterion(out, one_hot) + 0.1 * d
            else:
                if mode in ['single_CBM', 'single_vanilla', 'anchors']:
                    concepts_gnn, concepts_tab, out_t, out_img = model(input_text, input_image, y)
                    one_hot = torch.nn.functional.one_hot(y, num_classes=2).type_as(out_t)
                    loss = criterion(out_t, one_hot) + criterion(out_img, one_hot)
                    d = 0
                else:
                    concepts_gnn, concepts_tab, out, d = model(input_text, input_image, y)
                    one_hot = torch.nn.functional.one_hot(y, num_classes=2).type_as(out)
                    loss = criterion(out, one_hot)

            # calculate loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dist += d
            num_batches += 1

            optimizer.step()

        # get accuracy
        if mode in ['single_CBM', 'single_vanilla', 'anchors']:
            train_acc_g, train_acc_t = test_graph_class(model, train_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            test_acc_g, test_acc_t = test_graph_class(model, test_loader, if_interpretable_model=if_interpretable_model, mode=mode)
            train_acc = train_acc_g
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
        for input_image, input_text, _, _, y, _, _, _ in test_loader:
            # data.to(device)
            input_image = input_image.to(device)
            input_text = input_text.to(device)
            y = y.to(device)
            if if_interpretable_model:
                concepts_gnn, concepts_tab, out, _ = model(input_text, input_image, y)
            else:
                concepts_gnn, concepts_tab, out, _ = model(input_text, input_image, y)

            one_hot = torch.nn.functional.one_hot(y, num_classes=2).type_as(out)

            test_running_loss += criterion(out, one_hot).item()
            test_num_batches += 1

        train_loss.append(running_loss / num_batches)
        train_d.append(running_dist / num_batches)
        test_loss.append(test_running_loss / test_num_batches)

        if mode == 'SHARCS':
            model.epoch += 1
            if model.epoch > model.threshold:
                for param in model.conv.fc.parameters():
                    param.requires_grad = False
                for param in model.conv_linear.parameters():
                    param.requires_grad = False
                for param in model.text_encoder.parameters():
                    param.requires_grad = False

        if mode in ['single_CBM', 'single_vanilla', 'anchors']:
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

def plot_samples(clustering_model, data, num_nodes_view, questions, all_concepts, path, concepts=None, task='shared', mod='text'):

    res_sorted = clustering_utils.get_node_distances(clustering_model, data, concepts)
    sample_graphs = []
    sample_feat = []

    if isinstance(num_nodes_view, int):
        num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    unique_c_unfiltered, counts = np.unique((all_concepts>0.5) + 0, axis=0, return_counts=True)
    counts_filter = counts >= 0
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

    if mod=='image':
        fig, axes = plt.subplots(k, col, figsize=(18, 3 * k + 2))
        fig.suptitle(f'Nearest Instances to Cluster Centroid for {task} Concepts', y=1.005)

    top_concepts = []
    top_plot = []
    to_print = []
    for index, i in enumerate(l):
        distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        text = []
        image = []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            if mod=='text':
                text += [questions[idx] for idx in top_indices]
            if mod=='image':
                image += [questions[idx] for idx in top_indices]
        raw_concepts = data[top_indices][0]
        if mod == 'text':
            print('-------')
            counter = 0
            for q in text:
                print(q)
                if counter == 0:
                    top_concepts += [raw_concepts]
                    fig_aux = plt.figure(figsize=(8, 3))
                    txt = q
                    plt.ioff()
                    _ = plt.text(0, 0.5, f'{txt}', fontsize=35)
                    plt.axis('off')
                    fig_aux.savefig(f"./images/clevr_{i}_txt.png")
                    im = matplotlib.image.imread(f"./images/clevr_{i}_txt.png")
                    top_plot += [im]
                    plt.close()
                counter += 1
            print('-------')
        elif mod == 'image':
            counter = 1
            counter2 = 0
            for im_idx in image:
                if counter2 == 0:
                    top_concepts += [raw_concepts]
                    to_print += [im_idx]
                fig.add_subplot(k, col, index*col+counter)
                name = str(im_idx)
                name = '0'*(6-len(name)) + name
                fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
                img = np.array(Image.open(fname))
                plt.ioff()
                _ = plt.imshow(img.squeeze())
                try:
                    axes[index, counter-1].axis('off')
                except Exception as e:
                    axes[counter - 1].axis('off')
                plt.axis('off')
                counter += 1
                plt.close()
    plt.close('all')


    plt.savefig(os.path.join(path, f"{task}_g_concepts.pdf"))
    plt.savefig(os.path.join(path, f"{task}_g_concepts.png"))
    wandb.log({task: wandb.Image(plt)})
    plt.show()

    if mod == 'image':
        for index, el in enumerate(to_print):
            fig_aux, ax_aux = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))  # create figure & 1 axis
            name = str(el)
            name = '0' * (6 - len(name)) + name
            fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
            img = np.array(Image.open(fname))
            plt.ioff()
            _ = plt.imshow(img.squeeze())
            plt.axis("off")
            plt.savefig(f"./images/clevr_{index}_image.png")
            im = matplotlib.image.imread(f"./images/clevr_{index}_image.png")
            top_plot += [im]
            plt.close()
    plt.close('all')

    return top_plot, top_concepts

def save_centroids(centroids, y, used_centroid_labels, union_concepts,
                           g_concepts, questions,
                           t_concepts, idx,
                           path):


    res_sorted = clustering_utils.get_node_distances(None, union_concepts, centroids)
    unique_concepts = np.unique(used_centroid_labels)
    for c in tqdm(unique_concepts):
        distances = res_sorted[:, c]
        top_indices = np.argsort(distances)[::][0]
        text = questions[top_indices]
        image_index = idx[top_indices]

        plt.figure(figsize=(3, 3))
        plt.title(text, fontsize=10)
        name = str(image_index)
        name = '0' * (6 - len(name)) + name
        fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
        img = np.array(Image.open(fname))
        plt.ioff()
        _ = plt.imshow(img.squeeze())
        plt.axis('off')

        plt.savefig(f'{path}/{c}.svg')
        plt.close()
    plt.close('all')

def plot_combined_samples(clustering_model, data, y, num_nodes_view, questions, images_idx, images, all_concepts, path, concepts=None, task='combined'):

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

    unique_g_c = np.unique((data>0.5) + 0, axis=0).tolist()
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



    for i in l:
        distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        text = []
        image_index = []
        y_selected = []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            text += [questions[idx] for idx in top_indices]
            image_index += [images_idx[idx] for idx in top_indices]
            y_selected += [y[idx] for idx in top_indices]

        counter = 0
        for q, im_idx, y_obj in zip(text, image_index, y_selected):
            fig.add_subplot(k, col, i*col+1+counter)
            name = str(im_idx)
            name = '0'*(6-len(name)) + name
            fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
            img = np.array(Image.open(fname))
            plt.imshow(img.squeeze())
            plt.axis('off')
            axes[i,counter].set_title(f"{q}, label {y_obj}", fontsize=8)
            axes[i,counter].axis('off')
            counter += 1

    plt.show()
    plt.savefig(os.path.join(path, f"{task}_combined_concepts.pdf"))
    plt.savefig(os.path.join(path, f"{task}_combined_concepts.png"))


    return sample_graphs, sample_feat

def print_near_example(t_concepts, questions, g_concepts, img_idx, path, times=2, example=4):
    d = nn.PairwiseDistance(p=2)
    text_indexes = []
    image_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(t_concepts), size=(1,)).item()
        dist = d(torch.Tensor(g_concepts), torch.Tensor(t_concepts[sample_idx])).squeeze(-1)
        g_index = torch.argsort(dist)[:example]
        text_indexes += [sample_idx]
        image_indexes += [list(g_index.numpy())]
    figure = plt.figure(figsize=(22, 3 * times + 2))
    cols, rows = 5, times
    for i in range(1, times + 1):
        figure.add_subplot(rows, cols, (i-1)*(example+1)+1)
        plt.axis("off")
        q = (questions[text_indexes[i-1]].split(' ')[0] + '\n' +
             questions[text_indexes[i-1]].split(' ')[1] + '\n' +
             questions[text_indexes[i-1]].split(' ')[2] + '\n' +
             questions[text_indexes[i-1]].split(' ')[3])
        plt.text(0, 0.5, f'{q}', fontsize=25)
        for j, idx in enumerate(image_indexes[i - 1]):
            im_idx = img_idx[idx]
            name = str(im_idx)
            name = '0'*(6-len(name)) + name
            fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
            img = np.array(Image.open(fname))
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            plt.imshow(img.squeeze())
            plt.axis('off')
    plt.savefig(os.path.join(path, f"similar_txt.pdf"))
    plt.savefig(os.path.join(path, f"similar_txt.png"))
    wandb.log({'similar_img': wandb.Image(plt)})
    plt.show()

    figure = plt.figure(figsize=(20, 3 * times + 2))
    text_indexes = []
    image_indexes = []
    for i in range(times):
        sample_idx = torch.randint(len(t_concepts), size=(1,)).item()
        dist = d(torch.Tensor(t_concepts), torch.Tensor(g_concepts[sample_idx])).squeeze(-1)
        text_index = torch.argsort(dist)[:example]
        text_indexes += [list(text_index.numpy())]
        image_indexes += [sample_idx]
    cols, rows = 5, times
    for i in range(1, times + 1):
        figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 1)
        im_idx = img_idx[image_indexes[i-1]]
        name = str(im_idx)
        name = '0' * (6 - len(name)) + name
        fname = f'./clevr_data/train_full/CLEVR_train_full_{name}.png'
        img = np.array(Image.open(fname))
        plt.imshow(img.squeeze())
        plt.axis("off")
        for j, idx in enumerate(text_indexes[i - 1]):
            figure.add_subplot(rows, cols, (i - 1) * (example + 1) + 2 + j)
            q = (questions[idx].split(' ')[0] + '\n' +
                 questions[idx].split(' ')[1] + '\n' +
                 questions[idx].split(' ')[2] + '\n' +
                 questions[idx].split(' ')[3])
            plt.text(0, 0.5, f'{q}', fontsize=25)
            plt.axis('off')
    plt.savefig(os.path.join(path, f"similar_img.pdf"))
    plt.savefig(os.path.join(path, f"similar_img.png"))
    wandb.log({'similar_img': wandb.Image(plt)})
    plt.show()

def test_retrieval(dataloader, model, t_concepts, questions, img_concepts, txt_aux, anchors=False, anchors_txt=None, anchors_img=None):
    d = nn.PairwiseDistance(p=2)

    device = torch.device(dev)

    correct_shape_img = 0
    correct_size_img = 0
    correct_color_img = 0
    correct_material_img = 0
    correct_shape_txt = 0
    correct_size_txt = 0
    correct_color_txt = 0
    correct_material_txt = 0

    for _, (input_image, input_text, captions, _, y, _, sentence_aux, captions_aux) in enumerate(tqdm(dataloader)):
        filter = (sentence_aux != (torch.zeros(input_text[0].shape) - 1)).all(dim=1)
        sentence_aux = sentence_aux[filter]
        input_text = input_text[filter]
        input_image = input_image[filter]
        y = y[filter]
        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        text_rep, img_rep, _, _ = model(input_text, input_image, y)
        text_rep = text_rep.detach().cpu()
        img_rep = img_rep.detach().cpu()
        if anchors:
            img_rep = compute_relative_rep(img_rep, anchors_img)
            text_rep = compute_relative_rep(text_rep, anchors_txt)
        dist_text = []
        dist_img = []
        for img_c, txt_c in zip(img_concepts, t_concepts):
            d_text = d(text_rep, torch.Tensor(img_c).repeat(text_rep.shape[0],1))
            dist_text += [d_text]

            d_img = d(img_rep, torch.Tensor(txt_c).repeat(text_rep.shape[0], 1))
            dist_img += [d_img]
        dist_text = torch.vstack(dist_text)
        dist_img = torch.vstack(dist_img)

        txt_index = torch.argsort(dist_text, dim=0)[0]
        img_index = torch.argsort(dist_img, dim=0)[0]

        to_subtract = 0
        for c, txt_idx, c_aux, img_idx in zip(captions, txt_index, captions_aux, img_index):
            c = c.split(' ')
            c_tmp = questions[txt_idx].split(' ')
            if (c[0] == c_tmp[0]) or (synonyms[c[0]] == synonyms[c_tmp[0]]):
                correct_size_txt += 1
            if (c[1] == c_tmp[1]) or (synonyms[c[1]] == synonyms[c_tmp[1]]):
                correct_color_txt += 1
            if (c[2] == c_tmp[2]) or (synonyms[c[2]] == synonyms[c_tmp[2]]):
                correct_material_txt += 1
            if (c[3] == c_tmp[3]) or (synonyms[c[3]] == synonyms[c_tmp[3]]):
                correct_shape_txt += 1

            c_aux = c_aux.split(' ')
            if c_aux[0] == 'none':
                to_subtract += 1
                print('Missing aux_txt')
                continue
            c_aux_tmp = txt_aux[img_idx].split(' ')
            if (c_aux[0] == c_aux_tmp[0]) or (synonyms[c_aux[0]] == synonyms[c_aux_tmp[0]]):
                correct_size_img += 1
            if (c_aux[1] == c_aux_tmp[1]) or (synonyms[c_aux[1]] == synonyms[c_aux_tmp[1]]):
                correct_color_img += 1
            if (c_aux[2] == c_aux_tmp[2]) or (synonyms[c_aux[2]] == synonyms[c_aux_tmp[2]]):
                correct_material_img += 1
            if (c_aux[3] == c_aux_tmp[3]) or (synonyms[c_aux[3]] == synonyms[c_aux_tmp[3]]):
                correct_shape_img += 1

    sum_txt = correct_size_txt + correct_shape_txt + correct_color_txt + correct_material_txt
    sum_img = correct_size_img + correct_shape_img + correct_color_img + correct_material_img

    wandb.log({'txt_size': correct_size_txt/(len(dataloader.dataset)),
               'txt_shape': correct_shape_txt/(len(dataloader.dataset)),
               'txt_color': correct_color_txt/(len(dataloader.dataset)),
               'txt_material': correct_material_txt/(len(dataloader.dataset)),
               'img_size': correct_size_img/(len(dataloader.dataset) - to_subtract),
               'img_shape': correct_shape_img/(len(dataloader.dataset) - to_subtract),
               'img_color': correct_color_img/(len(dataloader.dataset) - to_subtract),
               'img_material': correct_material_img/(len(dataloader.dataset) - to_subtract),
               'sum_txt': sum_txt/(len(dataloader.dataset) * 4),
               'sum_img': sum_img/((len(dataloader.dataset) - to_subtract)*4)})

    return sum_txt/(len(dataloader.dataset) * 4), sum_img/(len(dataloader.dataset) * 4)

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

def collect_hidden_representation(model, dataloader):
    model.eval()
    device = torch.device(dev)
    img = []
    txt = []
    y_list = []
    for input_image, input_text, _, _, y, _, _, _ in dataloader:
        # data.to(device)
        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        tmp_txt, tmp_img, _, _ = model(input_text, input_image, y)
        txt += [tmp_txt.detach().cpu()]
        img += [tmp_img.detach().cpu()]
        y_list += [y.detach().cpu()]
    img = torch.vstack(img)
    txt = torch.vstack(txt)
    y_list = torch.hstack(y_list)
    return txt, img, y_list

def choose_anchors(model, dataloader):
    device = torch.device(dev)
    txt = []
    img = []
    counter = 0
    for input_image, input_text, _, _, y, _, _, _ in dataloader:

        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)

        tmp_txt, tmp_img, _, _ = model(input_text, input_image, y)

        filter = y == 1
        tmp_txt = tmp_txt[filter].detach().cpu()
        tmp_img = tmp_img[filter].detach().cpu()

        txt += [tmp_txt]
        img += [tmp_img]

        counter += tmp_txt.shape[0]

        if counter >= y.shape[0]:
            break

    img = torch.vstack(img)[:y.shape[0]]
    txt = torch.vstack(txt)[:y.shape[0]]

    return txt, img

def compute_relative_rep(mod, anchors):
    result = torch.Tensor()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for t in anchors:
        tmp = cos(mod, t).unsqueeze(dim=-1)
        result = torch.cat((result, tmp), dim=-1)
    return result

def test_missing_modality(dataloader, model, concepts_mod1, concepts_mod2):
    device = torch.device(dev)
    correct = 0
    for _, input_text, _, _, y, _, sentence_aux, _ in dataloader:
        filter = (sentence_aux != (torch.zeros(input_text[0].shape)-1)).all(dim=1)
        sentence_aux = sentence_aux[filter]
        input_text = input_text[filter]
        y = y[filter]
        input_image = sentence_aux.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        text, text_aux = model(input_text, input_image, y, missing=True, mod1=True)
        retreived_image = retreived_similar(text_aux.detach().cpu(), concepts_mod1)
        retreived_image = retreived_image.to(device)
        out = model(text, retreived_image, y, missing=True, prediction=True)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for input_image, _, _, _, y, img_aux, _, _ in dataloader:
        filter = torch.flatten((img_aux != (torch.zeros(input_image[0].shape)-1)), start_dim=1).all(dim=-1)
        img_aux = img_aux[filter]
        input_image = input_image[filter]
        y = y[filter]
        input_image = input_image.to(device)
        input_text = img_aux.to(device)
        y = y.to(device)
        img, img_aux = model(input_text, input_image, y, missing=True, mod2=True)
        retreived_text = retreived_similar(img_aux.detach().cpu(), concepts_mod2)
        retreived_text = retreived_text.to(device)
        out = model(retreived_text, img, y, missing=True, prediction=True)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    acc2 = correct / len(dataloader.dataset)
    return acc1, acc2

def test_missing_modality_anchors(dataloader, model, pred_model, concepts_mod1, concepts_mod2, anchors_mod1, anchors_mod2):
    device = torch.device(dev)
    correct = 0
    for _, input_text, _, _, y, _, sentence_aux, _ in dataloader:
        filter = (sentence_aux != (torch.zeros(input_text[0].shape) - 1)).all(dim=1)
        sentence_aux = sentence_aux[filter]
        input_text = input_text[filter]
        y = y[filter]
        input_image = sentence_aux.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        text, text_aux = model(input_text, input_image, y, missing=True, mod1=True)
        text_aux = compute_relative_rep(text_aux.detach().cpu(), anchors_mod2)
        text = compute_relative_rep(text.detach().cpu(), anchors_mod2)
        retreived_image = retreived_similar(text_aux.detach().cpu(), concepts_mod1)
        retreived_image = retreived_image.to(device)
        out = pred_model(text.to(device), retreived_image)
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    acc1 = correct / len(dataloader.dataset)
    correct = 0
    print('----MISSING MOD 2----')
    for input_image, _, _, _, y, img_aux, _, _ in dataloader:
        filter = torch.flatten((img_aux != (torch.zeros(input_image[0].shape) - 1)), start_dim=1).all(dim=-1)
        img_aux = img_aux[filter]
        input_image = input_image[filter]
        y = y[filter]
        input_image = input_image.to(device)
        input_text = img_aux.to(device)
        y = y.to(device)
        img, img_aux = model(input_text, input_image, y, missing=True, mod2=True)
        img_aux = compute_relative_rep(img_aux.detach().cpu(), anchors_mod1)
        img = compute_relative_rep(img.detach().cpu(), anchors_mod1)
        retreived_text = retreived_similar(img_aux.detach().cpu(), concepts_mod2)
        retreived_text = retreived_text.to(device)
        out = pred_model(retreived_text, img.to(device))
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
    acc2 = correct / len(dataloader.dataset)
    return acc1, acc2

def test_with_incremental_noise(model, dataloader, if_interpretable_model=True, mode='sharcs'):
    correct = 0
    correct_t = 0
    device = torch.device(dev)
    for input_image, input_text, _, _, y, _, _, _ in dataloader:
        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn, concepts_tab, out, _, _, _ = model(input_text, input_image, train_y, noise="mod1")
                concepts_gnn, concepts_tab, out2, _, _, _ = model(input_text, input_image, train_y, noise="mod2")
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        pred_t = out2.argmax(dim=1)
        correct_t += int((pred_t == y).sum())

    return correct / len(dataloader.dataset), correct_t / len(dataloader.dataset)

def test_with_interventions(model, n_concepts, dataloader, concepts_mod1, concepts_mod2, if_interpretable_model=True, mode='sharcs'):
    device = torch.device(dev)
    correct = 0
    correct_m= 0
    correct2 = 0
    correct_m2= 0
    p = int(n_concepts/2)
    for input_image, input_text, _, _, y, image_aux, sentence_aux, _ in dataloader:
        input_image = input_image.to(device)
        input_text = input_text.to(device)
        y = y.to(device)
        if if_interpretable_model:
            if mode != 'single_CBM':
                concepts_gnn_noise, concepts_tab_noise, out, _, _, _ = model(input_text, input_image, train_y, noise="mod2")
                concepts_gnn_noise2, concepts_tab_noise2, out2, _, _, _ = model(input_text, input_image, train_y, noise="mod1")
                concepts_gnn, concepts_tab, out, _, _, _ = model(input_text, input_image, train_y)
                concepts_noise = torch.cat([concepts_gnn_noise, concepts_tab_noise], dim=-1)
                concepts_noise2 = torch.cat([concepts_gnn_noise2, concepts_tab_noise2], dim=-1)
                concepts = torch.cat([concepts_gnn, concepts_tab], dim=-1)
                
                filter = (sentence_aux != (torch.zeros(input_text[0].shape)-1)).all(dim=1)
                sentence_aux_mod1 = sentence_aux[filter]
                input_text_mod1 = input_text[filter]
                y_mod1 = y[filter]
                input_image_mod1 = sentence_aux_mod1.to(device)
                input_text_mod1 = input_text_mod1.to(device)
                y_mod1 = y_mod1.to(device)
                text, text_aux = model(input_text_mod1, input_image_mod1, y_mod1, missing=True, mod1=True)
                retreived_image = retreived_similar(text_aux.detach().cpu(), concepts_mod2)
                retreived_image = retreived_image.to(device)
                concepts_retrieved1 = torch.cat([text, retreived_image], dim=-1)
                concepts_noise_missing_mod1 = concepts_noise.clone()

                filter = torch.flatten((img_aux != (torch.zeros(input_image[0].shape)-1)), start_dim=1).all(dim=-1)
                img_aux_mod2 = img_aux[filter]
                input_image_mod2 = input_image[filter]
                y_mod2 = y[filter]
                input_image_mod2 = input_image_mod2.to(device)
                input_text_mod2 = img_aux_mod2.to(device)
                y_mod2 = y_mod2.to(device)
                img, img_aux = model(input_text_mod2, input_image_mod2, y_mod2, missing=True, mod2=True)
                retreived_text = retreived_similar(img_aux.detach().cpu(), concepts_mod1)
                retreived_text = retreived_text.to(device)
                concepts_retrieved2 = torch.cat([retreived_text, img], dim=-1)
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
                concepts_gnn, concepts_tab, out, _ = model(g_concepts, t_concepts, y, missing=True, prediction=True)
                g_concepts = concepts_noise2[:, :int(concepts_noise2.shape[1] / 2)]
                t_concepts = concepts_noise2[:, int(concepts_noise2.shape[1] / 2):]
                concepts_gnn, concepts_tab, out2, _ = model(g_concepts, t_concepts, y, missing=True, prediction=True)
                
                g_concepts_missing_mod = concepts_noise_missing_mod1[:, :int(concepts_noise_missing_mod1.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod1[:, int(concepts_noise_missing_mod1.shape[1] / 2):]
                concepts_gnn_missing_mod, concepts_tab_missing_mod, out_missing_mod, _ = model(g_concepts_missing_mod, t_concepts_missing_mod, y_mod1, missing=True, prediction=True)
                g_concepts_missing_mod = concepts_noise_missing_mod2[:, :int(concepts_noise_missing_mod2.shape[1] / 2)]
                t_concepts_missing_mod = concepts_noise_missing_mod2[:, int(concepts_noise_missing_mod2.shape[1] / 2):]
                concepts_gnn_missing_mod, concepts_tab_missing_mod, out_missing_mod2, _ = model(g_concepts_missing_mod, t_concepts_missing_mod, y_mod2, missing=True, prediction=True)
            else:
                raise Exception('Not implemented')
        else:
            raise Exception('Not implemented')
        pred = out.argmax(dim=1)
        correct += int((pred == y).sum())
        pred_missing = out_missing_mod.argmax(dim=1)
        correct_m += int((pred_missing == y_mod1).sum())

        pred2 = out2.argmax(dim=1)
        correct2 += int((pred2 == y).sum())
        pred_missing2 = out_missing_mod2.argmax(dim=1)
        correct_m2 += int((pred_missing2 == y_mod2).sum())

    return correct / len(dataloader.dataset), correct_m / len(dataloader.dataset), correct2 / len(dataloader.dataset), correct_m2 / len(dataloader.dataset)


def main():
    tag = 'clevr'
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

    TRAIN_TEST_SPLIT = 0.8

    VOCAB_DIM = wandb.config.vocab_dim['values']
    CLUSTER_ENCODING_SIZE = wandb.config.k['values']
    EPOCHS = wandb.config.epochs['values']

    LR = wandb.config.lr['values']

    BATCH_SIZE = wandb.config.batch_size['values']

    global dev
    dev = wandb.config.dev['values']
    device = torch.device(dev)

    LAYER_NUM = 3

    config = {'seed': seed,
                       'dataset_name': DATASET_NAME,
                       'model_name': MODEL_NAME,
                       'num_classes': NUM_CLASSES,
                       'train_test_split': TRAIN_TEST_SPLIT,
                       'batch_size': BATCH_SIZE,
                       'cluster_encoding_size': CLUSTER_ENCODING_SIZE,
                       'epochs': EPOCHS,
                       'lr': LR,
                      }
    persistence_utils.persist_experiment(config, path, 'config.z')

    # load data
    print("Reading dataset")

    train_loader, train, test_loader, test = data_utils.create_clevr()

    full_train_loader = DataLoader(train, batch_size=int(len(train) * 0.1), shuffle=True)
    full_test_loader = DataLoader(test, batch_size=int(len(test) * 0.1))

    print('Done!')

    # model training
    if MODE == 'vanilla':
        model = models.CLEVR_Vanilla(VOCAB_DIM, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_vanilla' or MODE == 'anchors':
        model = models.CLEVR_SingleVanilla(VOCAB_DIM, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = False
    elif MODE == 'single_CBM':
        model = models.CLEVR_SingleCBM(VOCAB_DIM, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True
    elif MODE == 'multi_CBM':
        model = models.CLEVR_MultiCBM(VOCAB_DIM, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True
    else:
        model = models.CLEVR_SHARCS(VOCAB_DIM, CLUSTER_ENCODING_SIZE, NUM_CLASSES)
        interpretable = True
        EPOCHS += 20
    model.to(dev)

    model_to_return = model
    # register hooks to track activation

    model = model_utils.register_hooks(model)

    # train
    train_acc, test_acc, train_loss, test_loss = train_graph_class(model, train_loader, test_loader, EPOCHS, LR, if_interpretable_model=interpretable,
                                                                  mode=MODE)
    persistence_utils.persist_model(model, path, 'model.z')

    if MODE == 'anchors':
        train_txt_representation, train_image_representation, train_y = collect_hidden_representation(model,
                                                                                                      train_loader)
        test_txt_representation, test_image_representation, test_y = collect_hidden_representation(model, test_loader)

        train_loader = DataLoader(train, batch_size=int(len(train) * 0.1), shuffle=True)

        anchors_txt, anchors_img = choose_anchors(model, train_loader)
        train_relative_txt = compute_relative_rep(train_txt_representation, anchors_txt)
        train_relative_img = compute_relative_rep(train_image_representation, anchors_img)
        test_relative_txt = compute_relative_rep(test_txt_representation, anchors_txt)
        test_relative_img = compute_relative_rep(test_image_representation, anchors_img)

        train_set = data_utils.Anchors(train_relative_txt, train_relative_img, train_y)
        test_set = data_utils.Anchors(test_relative_txt, test_relative_img, test_y)

        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

        input_size = train_set[0][0].shape[0] + train_set[0][1].shape[0]

        pred_model = models.PredModel(input_size, input_size * 3, NUM_CLASSES)
        pred_model.to(device)

        model_utils.train_prediction(pred_model, train_loader, test_loader, 20, LR, NUM_CLASSES, dev)

        print("\n_____________THIS IS FOR GRAPHS AND TABLE____________")

        train_input_image, train_input_text, train_questions, train_img_index, train_y, _, _, train_txt_aux = next(iter(full_train_loader))
        train_input_image = train_input_image.to(device)
        train_input_text = train_input_text.to(device)
        train_y = train_y.to(device)

        train_txt, train_img, _, _ = model(train_input_text, train_input_image, train_y)

        train_txt = train_txt.cpu()
        train_img = train_img.cpu()

        train_txt = compute_relative_rep(train_txt, anchors_txt)
        train_img = compute_relative_rep(train_img, anchors_img)

        test_input_image, test_input_text, test_questions, test_img_index, test_y, _, _, test_txt_aux = next(iter(full_test_loader))
        test_input_image = test_input_image.to(device)
        test_input_text = test_input_text.to(device)
        test_y = test_y.to(device)

        test_txt, test_img, _, _ = model(test_input_text, test_input_image, test_y)
        test_txt = test_txt.cpu()
        test_img = test_img.cpu()

        test_txt = compute_relative_rep(test_txt, anchors_txt)
        test_img = compute_relative_rep(test_img, anchors_img)

        img_concepts = torch.vstack([train_img, test_img])
        txt_concepts = torch.vstack([train_txt, test_txt])

        q_train = train_input_text.cpu()
        q_test = test_input_text.cpu()
        q = torch.cat((q_train, q_test))

        img_train = train_input_image.cpu()
        img_test = test_input_image.cpu()
        img = torch.cat((img_train, img_test))

        y_train = train_y.cpu()
        y_test = test_y.cpu()
        y = torch.cat((y_train, y_test))

        questions = train_questions + test_questions
        txt_aux = train_txt_aux + test_txt_aux
        idx = torch.cat((train_img_index, test_img_index)).numpy()

        train_mask = np.zeros(y.shape[0], dtype=bool)
        train_mask[:y_train.shape[0]] = True
        test_mask = ~train_mask

        concepts = torch.vstack([torch.Tensor(txt_concepts), torch.Tensor(img_concepts)]).detach()
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
        txt_concepts = txt_concepts.detach().cpu().numpy()
        img_concepts = img_concepts.detach().numpy()

        print_near_example(txt_concepts, questions, img_concepts, idx, path)
        missing_accuracy = test_missing_modality_anchors(full_test_loader, model, pred_model,
                                                         img_concepts, txt_concepts,
                                                         anchors_img, anchors_txt)
        wandb.log({'missing image modality accuracy': missing_accuracy[0],
                   'missing text modality accuracy': missing_accuracy[1]})
        print(missing_accuracy)
        retrieval_acc = test_retrieval(full_test_loader, model, txt_concepts, questions, img_concepts, txt_aux,
                                       anchors=True, anchors_txt=anchors_txt, anchors_img=anchors_img)
        print(retrieval_acc)


    if interpretable:
        train_input_image, train_input_text, train_questions, train_img_index, train_y, _, _, train_txt_aux = next(iter(full_train_loader))
        train_input_image = train_input_image.to(device)
        train_input_text = train_input_text.to(device)
        train_y = train_y.to(device)

        train_node_concepts, _, _, _ = model(train_input_text, train_input_image, train_y)
        train_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        train_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        train_tab_local_concepts = model.x_tab_local_concepts.cpu()
        train_tab_concepts = model.tab_shared_concepts.cpu()

        test_input_image, test_input_text, test_questions, test_img_index, test_y, _, _, test_txt_aux = next(iter(full_test_loader))
        test_input_image = test_input_image.to(device)
        test_input_text = test_input_text.to(device)
        test_y = test_y.to(device)
        test_node_concepts, _, _, _ = model(test_input_text, test_input_image, test_y)
        test_graph_concepts = model.gnn_graph_shared_concepts.cpu()
        test_graph_local_concepts = model.gnn_graph_local_concepts.cpu()
        test_tab_local_concepts = model.x_tab_local_concepts.cpu()
        test_tab_concepts = model.tab_shared_concepts.cpu()

        graph_concepts = torch.vstack([train_graph_concepts, test_graph_concepts])
        graph_local_concepts = torch.vstack([train_graph_local_concepts, test_graph_local_concepts])
        tab_local_concepts = torch.vstack([train_tab_local_concepts, test_tab_local_concepts])
        tab_concepts = torch.vstack([train_tab_concepts, test_tab_concepts])

        q_train = train_input_text.cpu()
        q_test = test_input_text.cpu()
        q = torch.cat((q_train, q_test))

        idx_train = train_img_index.cpu()
        idx_test = test_img_index.cpu()

        img_train = train_input_image.cpu()
        img_test = test_input_image.cpu()
        img = torch.cat((img_train, img_test))

        y_train = train_y.cpu()
        y_test = test_y.cpu()
        y = torch.cat((y_train, y_test))

        questions = train_questions + test_questions
        txt_aux = train_txt_aux + test_txt_aux
        idx = torch.cat((train_img_index, test_img_index)).numpy()

        train_mask = np.zeros(y.shape[0], dtype=bool)
        train_mask[:y_train.shape[0]] = True
        test_mask = ~train_mask

        persistence_utils.persist_experiment(graph_concepts, path, 'graph_concepts.z')
        persistence_utils.persist_experiment(tab_concepts, path, 'tab_concepts.z')

        print("\n_____________THIS IS FOR TEXT____________")
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
        # wandb.log({'local graph completeness': classifier.accuracy, 'num clusters local graph': len(centroids)})

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_g_local, y, centroids, centroid_labels, used_centroid_labels,
                                            MODEL_NAME, LAYER_NUM, path, task="graph local", id_path="_graph")

        g_concepts = concepts_g_local.detach().cpu().numpy()

        print('TEXT CONCEPTS')
        sample_graphs, sample_feat = plot_samples(None, g_concepts, 5, questions, concepts_g_local, path,
                                                  concepts=centroids, task='local')

        print("\n_____________THIS IS FOR IMAGE____________")
        concepts_g_img_local = torch.Tensor(tab_local_concepts).detach()
        # find centroids for both modalities
        centroids, centroid_labels, used_centroid_labels = clustering_utils.find_centroids(concepts_g_img_local, y)
        print(f"Number of graph cenroids: {len(centroids)}")
        persistence_utils.persist_experiment(centroids, path, 'centroids_g.z')
        persistence_utils.persist_experiment(centroid_labels, path, 'centroid_labels_g.z')
        persistence_utils.persist_experiment(used_centroid_labels, path, 'used_centroid_labels_g.z')

        # calculate cluster sizing
        cluster_counts = visualisation_utils.print_cluster_counts(used_centroid_labels)
        concepts_g_img_local_tree = (concepts_g_img_local.detach().numpy() > 0.5).astype(int)
        classifier = models.ActivationClassifierConcepts(y, concepts_g_img_local_tree, train_mask, test_mask)

        print(f"Classifier Concept completeness score: {classifier.accuracy}")
        concept_metrics = [('cluster_count', cluster_counts)]
        persistence_utils.persist_experiment(concept_metrics, path, 'image_concept_metrics.z')
        # wandb.log({'local graph completeness': classifier.accuracy, 'num clusters local graph': len(centroids)})

        # plot concept heatmaps
        # visualisation_utils.plot_concept_heatmap(centroids, concepts, y_double, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

        # plot clustering
        visualisation_utils.plot_clustering(seed, concepts_g_img_local, y, centroids, centroid_labels,
                                            used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="graph image local",
                                            id_path="_graph")
        g_img_concepts = concepts_g_img_local.detach().cpu().numpy()

        print('IMAGE CONCEPTS')
        sample_graphs, sample_feat = plot_samples(None, g_img_concepts, 5, idx, concepts_g_img_local, path,
                                                  concepts=centroids, task='local', mod='image')

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
            tab_or_graph = torch.cat((torch.ones(int(concepts.shape[0] / 2)), torch.zeros(int(concepts.shape[0] / 2))),
                                     dim=0)
            visualisation_utils.plot_clustering(seed, concepts, tab_or_graph, centroids, centroid_labels,
                                                used_centroid_labels, MODEL_NAME, LAYER_NUM, path, task="shared",
                                                id_path="_graph")
            print('TEXT CONCEPTS')
            g_concepts = graph_concepts.detach().cpu().numpy()
            top_plot, top_concepts = plot_samples(None, g_concepts, 5, questions, concepts, path, concepts=centroids, task='local')

            print('Image CONCEPTS')
            t_concepts = tab_concepts.detach().numpy()
            top_plot_images, top_concepts_img = plot_samples(None, t_concepts, 5, idx, concepts, path, concepts=centroids, task='global', mod='image')

            print('------SHARED SPACE-----')
            top_concepts_both = np.array(top_concepts + top_concepts_img)

            top_plot_both = top_plot + top_plot_images

            if len(top_concepts + top_concepts_img) > 0:
                visualisation_utils.plot_clustering_images_inside(seed, concepts, top_concepts_both, top_plot_both,
                                                                  used_centroid_labels, path,
                                                                  'shared space with images')

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
                classifier.plot2(path, [union_concepts.detach(), y, None, idx, questions, path], integer=3, mode='clevr')
            except Exception as e:
                print(e)
            classifier = models.ActivationClassifierConcepts(y, union_concepts_tree, train_mask, test_mask,
                                                             max_depth=5)

            print(f"Classifier Concept completeness score: {classifier.accuracy}")
            concept_metrics = [('cluster_count', cluster_counts)]
            persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
            wandb.log({'combined completeness 2': classifier.accuracy})
            try: 
                classifier.plot2(path, [union_concepts.detach(), y, None, idx, questions, path], integer=3, mode='clevr')
            except Exception as e:
                print(e)

            concept_metrics = [('cluster_count', cluster_counts)]
            persistence_utils.persist_experiment(concept_metrics, path, 'graph_concept_metrics.z')
            wandb.log({'combined completeness 2': classifier.accuracy})

            # plot concept heatmaps
            # visualisation_utils.plot_concept_heatmap(centroids, union_concepts, y, used_centroid_labels, MODEL_NAME, LAYER_NUM, path, id_title="Graph ", id_path="graph_")

            # plot clustering
            visualisation_utils.plot_clustering(seed, union_concepts, y, centroids, centroid_labels, used_centroid_labels,
                                                MODEL_NAME, LAYER_NUM, path, task="combined", id_path="_graph",
                                                extra=True, train_mask=train_mask, test_mask=test_mask, n_classes=NUM_CLASSES)

            print_near_example(g_concepts, questions, t_concepts, idx, path)
            missing_accuracy = test_missing_modality(full_test_loader, model, t_concepts, g_concepts)
            wandb.log({'missing image modality accuracy': missing_accuracy[0],
                       'missing text modality accuracy': missing_accuracy[1]})
            print(missing_accuracy)

            retrieval_acc = test_retrieval(full_test_loader, model, g_concepts, questions, t_concepts, txt_aux)
            print(retrieval_acc)

            plot_combined_samples(None, union_concepts, y, 5, questions, idx, None, union_concepts.detach().cpu().numpy(),
                              path, concepts=centroids)
    # clean up
    plt.close('all')

    return model_to_return

if __name__ == '__main__':
    main()
