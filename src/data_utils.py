import os
import torch
import random
import numpy as np

import networkx as nx

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.utils import add_random_edge
from networkx import betweenness_centrality
from torch_geometric.utils.convert import to_networkx
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
from torch_geometric.datasets import MNISTSuperpixels
import torch_geometric.transforms as T
from torch_geometric.utils import from_networkx, to_networkx, subgraph
import collections
import json
from PIL import Image
from torchvision import transforms
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader, DenseDataLoader

def relabel_batch(batch):
    new_batch = []
    real_label = 0
    prev_label = None

    for b in batch:
        if prev_label is None:
            prev_label = b
        elif b != prev_label:
            real_label += 1
            prev_label = b

        new_batch.append(real_label)

    return torch.from_numpy(np.array(new_batch))

def load_json(question_file_path):
    with open(question_file_path) as f:
        data = json.load(f)
    image_index = []
    question = []
    answer = []
    for el in data['questions']:
        image_index += [el['image_index']]
        question += [el['question']]
        answer += [el['answer']]
    return image_index, question, answer


def half_graph(graphs):
  cutted_graphs = []
  cutted_graphs_aux = []
  for g in graphs[:int(len(graphs)/2)]:
    max_c = (torch.transpose(g.pos, 1, 0).max(dim=-1)[0][1]/2).item()
    nodes_to_remove = []
    nodes_to_keep = []
    for i, n in enumerate(g.x):
      if g.pos[i][1].item() > max_c:
        nodes_to_remove += [i]
      else:
        nodes_to_keep += [i]
    nodes_to_keep = torch.tensor(nodes_to_keep)
    g_tmp = to_networkx(g)
    g_tmp.remove_nodes_from(nodes_to_remove)
    g_pyg = from_networkx(g_tmp)
    g_pyg.x = g.x[nodes_to_keep]
    g_pyg.y = g.y
    g_pyg.pos = g.pos[nodes_to_keep]
    g_pyg.edge_attr = subgraph(nodes_to_keep, g.edge_index, g.edge_attr)[1]
    cutted_graphs += [g_pyg]
  for g in graphs[int(len(graphs)/2):]:
    max_c = (torch.transpose(g.pos, 1, 0).max(dim=-1)[0][1]/2).item()
    nodes_to_remove = []
    nodes_to_keep = []
    for i, n in enumerate(g.x):
      if g.pos[i][1].item() <= max_c:
        nodes_to_remove += [i]
      else:
        nodes_to_keep += [i]
    nodes_to_keep = torch.tensor(nodes_to_keep)
    g_tmp = to_networkx(g)
    g_tmp.remove_nodes_from(nodes_to_remove)
    g_pyg = from_networkx(g_tmp)
    g_pyg.x = g.x[nodes_to_keep]
    g_pyg.y = g.y
    g_pyg.pos = g.pos[nodes_to_keep]
    g_pyg.edge_attr = subgraph(nodes_to_keep, g.edge_index, g.edge_attr)[1]
    cutted_graphs += [g_pyg]
  for g in graphs[:int(len(graphs)/2)]:
    max_c = (torch.transpose(g.pos, 1, 0).max(dim=-1)[0][1]/2).item()
    nodes_to_remove = []
    nodes_to_keep = []
    for i, n in enumerate(g.x):
      if g.pos[i][1].item() <= max_c:
        nodes_to_remove += [i]
      else:
        nodes_to_keep += [i]
    nodes_to_keep = torch.tensor(nodes_to_keep)
    g_tmp = to_networkx(g)
    g_tmp.remove_nodes_from(nodes_to_remove)
    g_pyg = from_networkx(g_tmp)
    g_pyg.x = g.x[nodes_to_keep]
    g_pyg.y = g.y
    g_pyg.pos = g.pos[nodes_to_keep]
    g_pyg.edge_attr = subgraph(nodes_to_keep, g.edge_index, g.edge_attr)[1]
    cutted_graphs_aux += [g_pyg]
  for g in graphs[int(len(graphs)/2):]:
    max_c = (torch.transpose(g.pos, 1, 0).max(dim=-1)[0][1]/2).item()
    nodes_to_remove = []
    nodes_to_keep = []
    for i, n in enumerate(g.x):
      if g.pos[i][1].item() > max_c:
        nodes_to_remove += [i]
      else:
        nodes_to_keep += [i]
    nodes_to_keep = torch.tensor(nodes_to_keep)
    g_tmp = to_networkx(g)
    g_tmp.remove_nodes_from(nodes_to_remove)
    g_pyg = from_networkx(g_tmp)
    g_pyg.x = g.x[nodes_to_keep]
    g_pyg.y = g.y
    g_pyg.pos = g.pos[nodes_to_keep]
    g_pyg.edge_attr = subgraph(nodes_to_keep, g.edge_index, g.edge_attr)[1]
    cutted_graphs_aux += [g_pyg]

  return cutted_graphs, cutted_graphs_aux

def create_halfmnist(size, batch_size, num_classes):
    dset = MNIST("./mnist", download=True)
    transform = T.Cartesian(cat=False)
    graphs = MNISTSuperpixels('./data/', True, transform=transform)[:size]
    graphs, graphs_aux = half_graph(graphs)

    # imgs = dataset['train'][:10000]['image']
    imgs = dset.data.unsqueeze(-1).numpy().astype(np.float64)[:size]
    imgs = np.array(list(map(np.array, imgs)))

    up = imgs[:, :14, :, :]
    down = imgs[:, 14:, :, :]
    imgs_cnn = np.concatenate((down[:int(imgs.shape[0] / 2)], up[int(imgs.shape[0] / 2):]), axis=0)
    imgs_aux = np.concatenate((up[:int(imgs.shape[0] / 2)], down[int(imgs.shape[0] / 2):]), axis=0)

    up_down = np.array([0] * int(imgs.shape[0] / 2) + [1] * int(imgs.shape[0] / 2))
    labels = dset.targets[:size].numpy()

    train_mask = np.zeros(size, dtype=bool)
    train_mask[:int(size*0.8)] = True
    test_mask = ~train_mask

    labels_tmp = torch.tensor([ele.y for i, ele in enumerate(graphs) if train_mask[i]]).long().squeeze(-1)
    y_train_one_hot = torch.nn.functional.one_hot(torch.Tensor(labels_tmp).long(), num_classes=num_classes)
    weights = ((y_train_one_hot.sum(dim=0) - labels_tmp.shape[0]) * (-1)) / y_train_one_hot.sum(dim=0)

    train = HALFMNIST(graphs, imgs_cnn, imgs_aux, graphs_aux, labels, up_down, train_mask)
    test = HALFMNIST(graphs, imgs_cnn, imgs_aux, graphs_aux, labels, up_down, test_mask)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, follow_batch=['x', 'x_image'])
    test_loader = DataLoader(test, batch_size=batch_size, follow_batch=['x', 'x_image'])

    return train_loader, train, test_loader, test, weights

def create_mnist_superpixels(size, batch_size, num_classes):
    dset = MNIST("./mnist", download=True)
    transform = T.Cartesian(cat=False)
    graphs = MNISTSuperpixels('./data/', True, transform=transform)[:size]
    graphs_aux = list(graphs)[1:size] + [list(graphs)[0]]
    imgs_aux = dset.data.unsqueeze(-1).numpy().astype(np.float64)[:size]

    imgs = np.concatenate((imgs_aux[1:size], np.expand_dims(imgs_aux[0], axis=0)), axis=0)
    imgs = np.array(list(map(np.array, imgs)))
    labels_img = dset.targets[:size].numpy()
    labels_img = np.concatenate((labels_img[1:size], np.expand_dims(labels_img[0], axis=0)), axis=0)

    train_mask = np.zeros(size, dtype=bool)
    train_mask[:int(size*0.8)] = True
    test_mask = ~train_mask

    labels = torch.tensor([ele.y for i, ele in enumerate(graphs) if train_mask[i]]).long().squeeze(-1)
    labels = labels + labels_img[train_mask]
    y_train_one_hot = torch.nn.functional.one_hot(torch.Tensor(labels).long(), num_classes=num_classes)
    weights = ((y_train_one_hot.sum(dim=0) - labels.shape[0]) * (-1)) / y_train_one_hot.sum(dim=0)

    train = MNIST_SUPERPIXELS2(graphs, imgs, imgs_aux, graphs_aux, labels_img, train_mask)
    test = MNIST_SUPERPIXELS2(graphs, imgs, imgs_aux, graphs_aux, labels_img, test_mask)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, follow_batch=['x', 'x_image'])
    test_loader = DataLoader(test, batch_size=batch_size, follow_batch=['x', 'x_image'])

    return train_loader, train, test_loader, test, weights

def retrieve_aux(images, questions, answers):
    images_aux = []
    questions_aux = []

    for idx, img, q, y in zip(list(range(len(images))), images, questions, answers):
        check_image = False
        check_text = False
        if y == 1:
            questions_aux += [q]
            check_text = True
            images_aux += [img]
            check_image = True
        for j in (list(range(len(images)))[idx+1:]+list(range(len(images)))[:idx+1]):
            if q == questions[j] and answers[j] == 1 and not check_image:
                images_aux += [images[j]]
                check_image = True
            if (img == images[j] and answers[j] == 1) and not check_text:
                questions_aux += [questions[j]]
                check_text = True
            if check_image and check_text:
                break
        if not check_image:
            images_aux += [-1]
        elif not check_text:
            questions_aux += ['none']

    return images_aux, questions_aux


def create_clevr():
    image_index, question, answer = load_json('./clevr_data/CLEVR_questions_full.json')
    size = int(len(answer) * 0.8)
    image_aux, caption_aux = retrieve_aux(image_index, question, answer)
    # clevr_train = CLEVR(question_string[:size], image_idxs[:size], answers[:size], './test/output/CLEVR_train_', train=True, vectorizer=None)
    clevr_train = CLEVR(question[:size], image_index[:size], answer[:size], image_aux[:size], caption_aux[:size],
                        './clevr_data/train_full/CLEVR_train_full_', train=True, vectorizer=None)
    train_loader = DataLoader(clevr_train, batch_size=100, shuffle=True)
    vectorizer = clevr_train.vectorizer
    # clevr_test = CLEVR(question_string[size:], image_idxs[size:], answers[size:], './test/output/CLEVR_train_', train=False, vectorizer=vectorizer)
    clevr_test = CLEVR(question[size:], image_index[size:], answer[size:], image_aux[size:], caption_aux[size:],
                       './clevr_data/train_full/CLEVR_train_full_',
                       train=False, vectorizer=vectorizer)
    test_loader = DataLoader(clevr_test, batch_size=100)
    return train_loader, clevr_train, test_loader, clevr_test

class PairData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'aux_edge_index':
            return self.x_tab.size(0)
        return super().__inc__(key, value, *args, **kwargs)

class PairDataMNIST(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return self.x.size(0)
        if key == 'edge_index_image':
            return self.x_image.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def create_xor_dataset(n):
    dataset = []
    for i in tqdm(range(n)):
        if random.random() > 0.5:
            f1 = 1
        else:
            f1 = 0
        if random.random() > 0.5:
            f2 = 1
        else:
            f2 = 0
        if random.random() > 0.5:
            f3 = 1
        else:
            f3 = 0
        if f2 == 1:
            g1_edge_list = [[], []]
            for i in range(4):
                g1_edge_list[0] += [i]
                g1_edge_list[1] += [i]

                g1_edge_list[0] += [i]
                g1_edge_list[1] += [(i+1)%4]

                g1_edge_list[0] += [i]
                g1_edge_list[1] += [(i-1)%4]
        else:
            g1_edge_list = [list(range(4)),list(range(4))]
        if f3 == 1:
            g2_edge_list = [[], []]
            for i in range(6):
                g2_edge_list[0] += [i+4]
                g2_edge_list[1] += [i+4]

                g2_edge_list[0] += [i+4]
                g2_edge_list[1] += [(i+1)%6+4]

                g2_edge_list[0] += [i+4]
                g2_edge_list[1] += [(i-1)%6+4]
        else:
            g2_edge_list = [list(range(4, 10)),list(range(4, 10))]
        if f1 == 1:
            g3_edge_list = [[], []]
            for i in range(6):
                g3_edge_list[0] += [i+4]
                g3_edge_list[1] += [i+4]

                g3_edge_list[0] += [i+4]
                g3_edge_list[1] += [(i+1)%6+4]

                g3_edge_list[0] += [i+4]
                g3_edge_list[1] += [(i-1)%6+4]
        else:
            g3_edge_list = [list(range(4, 10)),list(range(4, 10))]

        g1_edge_list = torch.Tensor(g1_edge_list)
        g2_edge_list = torch.Tensor(g2_edge_list)
        g3_edge_list = torch.Tensor(g3_edge_list)

        if f2 == 1 and f3 == 1:
            rand = (random.randint(0, 3), random.randint(4, 9))
            graph = torch.concat((g1_edge_list, g2_edge_list,
                    torch.Tensor([[rand[0], rand[1]], [rand[1], rand[0]]])), dim=-1).long()
        else:
            graph = torch.concat((g1_edge_list, g2_edge_list), dim=-1).long()

        if f2 == 1 and f1 == 1:
            rand = (random.randint(0, 3), random.randint(4, 9))
            graph_tab = torch.concat((g1_edge_list, g3_edge_list,
                    torch.Tensor([[rand[0], rand[1]], [rand[1], rand[0]]])), dim=-1).long()
        else:
            graph_tab = torch.concat((g1_edge_list, g3_edge_list), dim=-1).long()

        graph, added_edges = add_random_edge(graph, p=0.2,
                                          force_undirected=True)

        graph_tmp = Data(edge_index=graph, y=0)
        x = torch.Tensor(list(betweenness_centrality(to_networkx(graph_tmp)).values())).unsqueeze(dim=1)

        graph_tab, added_edges = add_random_edge(graph_tab, p=0.2,
                                             force_undirected=True)

        graph_tmp = Data(edge_index=graph_tab, y=0)
        x_tab = torch.Tensor(list(betweenness_centrality(to_networkx(graph_tmp)).values())).unsqueeze(dim=1)

        graph_stats = torch.tensor([random.randint(0, 1),
                                      f1, random.randint(0, 1),
                                      random.randint(0, 1),
                                      f2, random.randint(0, 1)
                                      ]).float()
        graph_aux = torch.tensor([random.randint(0, 1),
                                      f3, random.randint(0, 1),
                                      random.randint(0, 1),
                                      f2, random.randint(0, 1)
                                      ]).float()
        graph_stats_clean = torch.tensor([f1, f2, f3])
        y = int((f1 ^ f2) and (f2 ^ f3))
        if random.random() < 0.2:
            tab_anchor = 1
        else:
            tab_anchor = 0
        g = PairData(x=x, edge_index=graph, x_tab=x_tab, aux_edge_index=graph_tab,
                     graph_stats=graph_stats, graph_aux=graph_aux, graph_stats_clean=graph_stats_clean,
                     y=y, tab_anchor=tab_anchor)
        dataset += [g]
    return dataset



class Scale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)
        
class Anchors(Dataset):
    def __init__(self, mod1, mod2, y):
        super(Dataset, self).__init__()
        self.mod1 = mod1
        self.mod2 = mod2
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        mod1 = self.mod1[index]
        mod2 = self.mod2[index]
        y = self.y[index]

        return mod1, mod2, y

class CLEVR(Dataset):
    def __init__(self, captions, img_indexes, labels, img_aux, q_aux, path, train=True, vectorizer=None):
        super(Dataset, self).__init__()

        self.path = path
        self.corpus = captions
        if train:
            self.vectorizer = CountVectorizer()
            self.captions = torch.Tensor(self.vectorizer.fit_transform(self.corpus).toarray())
        else:
            self.vectorizer = vectorizer
            self.captions = torch.Tensor(self.vectorizer.transform(self.corpus).toarray())
        self.corpus_aux = q_aux
        self.captions_aux = torch.Tensor(self.vectorizer.transform(self.corpus_aux).toarray())
        self.img_aux = torch.Tensor(img_aux).long()
        self.labels = torch.Tensor(labels).long()
        self.img_indexes = torch.Tensor(img_indexes).long()
        self.preprocess = transforms.Compose([
            Scale([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        name = str(self.img_indexes[index].item())
        name = '0'*(6-len(name)) + name
        fname = f'{self.path}{name}.png'
        img = Image.open(fname).convert('RGB')
        image = self.preprocess(img)
        img_index = self.img_indexes[index]

        if self.img_aux[index].item() != -1:
            name = str(self.img_aux[index].item())
            name = '0' * (6 - len(name)) + name
            fname = f'{self.path}{name}.png'
            img = Image.open(fname).convert('RGB')
            image_aux = self.preprocess(img)
        else:
            image_aux = torch.zeros(image.shape)-1

        caption = self.captions[index]
        sentence = self.corpus[index]

        caption_aux = self.captions_aux[index]
        if caption_aux.sum().item() == 0:
            caption_aux = caption_aux - 1
        sentence_aux = self.corpus_aux[index]
        if sentence_aux == 'none':
            sentence_aux = 'none none none none'

        label = self.labels[index]

        return image, caption, sentence, img_index, label, image_aux, caption_aux, sentence_aux

class HALFMNIST(Dataset):
    def __init__(self, graphs, images, images_aux, graphs_aux, labels_img, up_down, mask):
        super(Dataset, self).__init__()
        self.graphs_aux = [ele for i, ele in enumerate(graphs_aux) if mask[i]]
        self.graphs = [ele for i, ele in enumerate(graphs) if mask[i]]
        self.images_aux = torch.Tensor(images_aux)[mask]
        self.images = torch.Tensor(images)[mask]
        self.labels_img = torch.Tensor(labels_img)[mask].long()
        self.up_down = torch.Tensor(up_down)[mask].long()

    def __len__(self):
        return self.labels_img.shape[0]

    def __getitem__(self, index):
        return PairDataMNIST(x=self.graphs[index].x,
                    pos=self.graphs[index].pos,
                    edge_index=self.graphs[index].edge_index,
                    edge_attr=self.graphs[index].edge_attr,
                    x_image=self.graphs_aux[index].x,
                    pos_image=self.graphs_aux[index].pos,
                    edge_index_image=self.graphs_aux[index].edge_index,
                    edge_attr_image=self.graphs_aux[index].edge_attr,
                    img=self.images[index].squeeze(-1).unsqueeze(0).unsqueeze(1),
                    img_aux=self.images_aux[index].squeeze(-1).unsqueeze(0).unsqueeze(1),
                    anchor=torch.Tensor([random.random() < 0.1]).bool(),
                    y=self.graphs[index].y,
                    y_img=self.labels_img[index],
                    y_g=self.graphs[index].y,
                    up_down=self.up_down[index])


class MNIST_SUPERPIXELS2(Dataset):
    def __init__(self, graphs, images, images_aux, graphs_aux, labels_img, mask, noise=False):
        super(Dataset, self).__init__()
        self.graphs_aux = [ele for i, ele in enumerate(graphs_aux) if mask[i]]
        self.graphs = [ele for i, ele in enumerate(graphs) if mask[i]]
        self.images_aux = torch.Tensor(images_aux)[mask]
        self.images = torch.Tensor(images)[mask]
        if noise:
            self.images_aux = self.images_aux + torch.normal(mean=0, std=128, size=self.images_aux.shape)
            self.images_aux = torch.clamp(self.images_aux, min=0, max=255)
            self.images = self.images + torch.normal(mean=0, std=128, size=self.images.shape)
            self.images = torch.clamp(self.images, min=0, max=255)
        self.labels_img = torch.Tensor(labels_img)[mask].long()

    def __len__(self):
        return self.labels_img.shape[0]

    def __getitem__(self, index):
        return PairDataMNIST(x=self.graphs[index].x,
                    pos=self.graphs[index].pos,
                    edge_index=self.graphs[index].edge_index,
                    edge_attr=self.graphs[index].edge_attr,
                    x_image=self.graphs_aux[index].x,
                    pos_image=self.graphs_aux[index].pos,
                    edge_index_image=self.graphs_aux[index].edge_index,
                    edge_attr_image=self.graphs_aux[index].edge_attr,
                    img=self.images[index].squeeze(-1).unsqueeze(0).unsqueeze(1),
                    img_aux=self.images_aux[index].squeeze(-1).unsqueeze(0).unsqueeze(1),
                    anchor=torch.Tensor([random.random() < 0.1]).bool(),
                    y=self.graphs[index].y + self.labels_img[index],
                    y_img=self.labels_img[index],
                    y_g=self.graphs[index].y)


def create_path(path):
    if not os.path.isdir(path):
        os.makedirs(path)
