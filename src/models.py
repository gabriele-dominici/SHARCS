import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import tree
import matplotlib.pyplot as plt
import tree_viz
from sklearn.tree import DecisionTreeClassifier
import dtreeviz
from torch_geometric.nn import (
    GCNConv,
    SplineConv,
    global_mean_pool,
    global_add_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.utils import normalized_cut
import torch_geometric.transforms as T
import os

import numpy as np
import pandas as pd
from pytorch_lightning.utilities.seed import seed_everything

import model_utils

import math
import random
import wandb

class Tab_Graph(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)


        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, clean_features):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(clean_features):
                    if (clean_features[i][0] == clean_features[j][2] and
                       clean_features[i][1] == clean_features[j][1] and j not in graph_indexes):
                        graph_indexes += [j]
                        tab_indexes += [i]
                        break

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x, edge_index, batch, x_tab, clean_features):

        x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)
        clean_features = clean_features.reshape(x_tab.shape[0], 3)

        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = x.squeeze()

        self.gnn_node_embedding = x

        x = F.gumbel_softmax(x, tau=1, hard=True)

        self.gnn_node_concepts = x

        x = self.sum_pool(x, batch)
        x = self.batch_norm_g_1(x)
        x = F.sigmoid(x*5)

        self.gnn_graph_local_concepts = x


        x = self.projection_graph(x)

        tab = self.mlp(x_tab)

        self.tab_activations = tab

        tab = self.batch_norm_img_1(tab)
        tab = F.sigmoid(tab*5)

        self.x_tab_local_concepts = tab

        tab = self.projection_tab(tab)

        concat = torch.cat((x, tab), dim=0)
        normalised = self.batch_norm_img_shared(concat)
        norm_concepts = F.sigmoid(normalised*5)

        x = norm_concepts[:int(norm_concepts.shape[0]/2)]
        tab = norm_concepts[int(norm_concepts.shape[0]/2):]

        d = self.dist(tab, x, clean_features)
        if math.isnan(d):
          d = 0

        self.gnn_graph_shared_concepts = x
        self.tab_shared_concepts = tab

        combined_concept = torch.cat((x, tab), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d


class Tab_Graph_missing(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_missing, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                              nn.ReLU(),
                                              nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                              )

        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                            nn.ReLU(),
                                            nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 10),
                                  nn.ReLU(),
                                  nn.Linear(10, num_classes)
                                  )

    def dist(self, x_tab, x_graph, clean_features):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(clean_features):
                    if (clean_features[i][0] == clean_features[j][2] and
                            clean_features[i][1] == clean_features[j][1] and j not in graph_indexes):
                        graph_indexes += [j]
                        tab_indexes += [i]
                        break

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x, edge_index, batch, x_tab, clean_features,
                missing=False, mod1=False, mod2=False, aux_edge=None, batch_aux=None, prediction=False):

        if not missing:
            x_tab = x_tab.reshape(batch[-1] + 1, self.num_in_features)
            clean_features = clean_features.reshape(x_tab.shape[0], 3)

            x = self.conv0(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv4(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv3(x, edge_index)
            x = F.leaky_relu(x)

            x = x.squeeze()

            self.gnn_node_embedding = x

            x = F.gumbel_softmax(x, tau=1, hard=True)

            self.gnn_node_concepts = x

            x = self.sum_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 5)

            self.gnn_graph_local_concepts = x

            x = self.projection_graph(x)

            tab = self.mlp(x_tab)

            self.tab_activations = tab

            tab = self.batch_norm_img_1(tab)
            tab = F.sigmoid(tab * 5)

            self.x_tab_local_concepts = tab

            tab = self.projection_tab(tab)

            concat = torch.cat((x, tab), dim=0)
            normalised = self.batch_norm_img_shared(concat)
            norm_concepts = F.sigmoid(normalised * 5)

            x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
            tab = norm_concepts[int(norm_concepts.shape[0] / 2):]

            d = self.dist(tab, x, clean_features)
            if math.isnan(d):
                d = 0

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = tab

            combined_concept = torch.cat((x, tab), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d
        elif prediction:
            combined_concept = torch.cat((x, x_tab), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, 0
        else:
            if mod1:
                x_tab = x_tab.reshape(batch[-1] + 1, self.num_in_features)
                tab = self.mlp(x_tab)
                tab = self.batch_norm_img_1(tab)
                tab = F.sigmoid(tab * 5)
                tab = self.projection_tab(tab)

                tab_aux = x.reshape(batch[-1] + 1, self.num_in_features)
                tab_aux = self.mlp(tab_aux)
                tab_aux = self.batch_norm_img_1(tab_aux)
                tab_aux = F.sigmoid(tab_aux * 5)
                tab_aux = self.projection_tab(tab_aux)

                concat = torch.cat((tab, tab_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                tab = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                tab_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]
                return tab, tab_aux
            elif mod2:
                x = self.conv0(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv1(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv2(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv4(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv3(x, edge_index)
                x = F.leaky_relu(x)
                x = x.squeeze()
                x = F.gumbel_softmax(x, tau=1, hard=True)
                x = self.sum_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)
                x = self.projection_graph(x)

                x_g_aux = self.conv0(x_tab, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv1(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv2(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv4(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv3(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = x_g_aux.squeeze()
                x_g_aux = F.gumbel_softmax(x_g_aux, tau=1, hard=True)
                x_g_aux = self.sum_pool(x_g_aux, batch_aux)
                x_g_aux = self.batch_norm_g_1(x_g_aux)
                x_g_aux = F.sigmoid(x_g_aux * 5)
                x_g_aux = self.projection_graph(x_g_aux)

                concat = torch.cat((x, x_g_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_g_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]
                return x, x_g_aux

class Tab_Graph_missing_extra(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_missing_extra, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)


        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, clean_features):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(clean_features):
                    if (clean_features[i][0] == clean_features[j][2] and
                       clean_features[i][1] == clean_features[j][1] and j not in graph_indexes):
                        graph_indexes += [j]
                        tab_indexes += [i]
                        break

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x, edge_index, batch, x_tab, clean_features,
                missing=False, mod1=False, mod2=False, aux_edge=None, batch_aux=None, prediction=False):

        if not missing:
            x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)
            clean_features = clean_features.reshape(x_tab.shape[0], 3)

            x = self.conv0(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv4(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv3(x, edge_index)
            x = F.leaky_relu(x)

            x = x.squeeze()

            self.gnn_node_embedding = x

            x = F.gumbel_softmax(x, tau=1, hard=True)

            self.gnn_node_concepts = x

            x = self.sum_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x*5)

            self.gnn_graph_local_concepts = x


            x = self.projection_graph(x)

            tab = self.mlp(x_tab)

            self.tab_activations = tab

            tab = self.batch_norm_img_1(tab)
            tab = F.sigmoid(tab*5)

            self.x_tab_local_concepts = tab

            tab = self.projection_tab(tab)

            concat = torch.cat((x, tab), dim=0)
            normalised = self.batch_norm_img_shared(concat)
            norm_concepts = F.sigmoid(normalised*5)

            x = norm_concepts[:int(norm_concepts.shape[0]/2)]
            tab = norm_concepts[int(norm_concepts.shape[0]/2):]

            d = self.dist(tab, x, clean_features)
            if math.isnan(d):
              d = 0

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = tab

            x = x.unsqueeze(1)
            tab = tab.unsqueeze(1)
            combined_concept = torch.cat((x, tab), dim=1)
            max_value = torch.max(combined_concept, dim=1)[0]
            min_value = torch.min(combined_concept, dim=1)[0]
            combined_concept = torch.cat((max_value, min_value), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d
        elif prediction:

            x = x.unsqueeze(1)
            x_tab = x_tab.unsqueeze(1)
            combined_concept = torch.cat((x, x_tab), dim=1)
            max_value = torch.max(combined_concept, dim=1)[0]
            min_value = torch.min(combined_concept, dim=1)[0]
            combined_concept = torch.cat((max_value, min_value), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, 0
        else:
            if mod1:
                x_tab = x_tab.reshape(batch[-1] + 1, self.num_in_features)
                tab = self.mlp(x_tab)
                tab = self.batch_norm_img_1(tab)
                tab = F.sigmoid(tab * 5)
                tab = self.projection_tab(tab)

                tab_aux = x.reshape(batch[-1] + 1, self.num_in_features)
                tab_aux = self.mlp(tab_aux)
                tab_aux = self.batch_norm_img_1(tab_aux)
                tab_aux = F.sigmoid(tab_aux * 5)
                tab_aux = self.projection_tab(tab_aux)

                concat = torch.cat((tab, tab_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                tab = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                tab_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]
                return tab, tab_aux
            elif mod2:
                x = self.conv0(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv1(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv2(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv4(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv3(x, edge_index)
                x = F.leaky_relu(x)
                x = x.squeeze()
                x = F.gumbel_softmax(x, tau=1, hard=True)
                x = self.sum_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)
                x = self.projection_graph(x)

                x_g_aux = self.conv0(x_tab, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv1(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv2(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv4(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv3(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = x_g_aux.squeeze()
                x_g_aux = F.gumbel_softmax(x_g_aux, tau=1, hard=True)
                x_g_aux = self.sum_pool(x_g_aux, batch_aux)
                x_g_aux = self.batch_norm_g_1(x_g_aux)
                x_g_aux = F.sigmoid(x_g_aux * 5)
                x_g_aux = self.projection_graph(x_g_aux)

                concat = torch.cat((x, x_g_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_g_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]
                return x, x_g_aux

class Tab_Graph_MultiCBM(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_MultiCBM, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)


        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, clean_features):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(clean_features):
                    if (clean_features[i][0] == clean_features[j][2] and
                       clean_features[i][1] == clean_features[j][1] and j not in graph_indexes):
                        graph_indexes += [j]
                        tab_indexes += [i]
                        break

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x, edge_index, batch, x_tab, clean_features,
                missing=False, mod1=False, mod2=False, aux_edge=None, batch_aux=None, prediction=False):

        if not missing:
            x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)
            clean_features = clean_features.reshape(x_tab.shape[0], 3)

            x = self.conv0(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv4(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv3(x, edge_index)
            x = F.leaky_relu(x)

            x = x.squeeze()

            self.gnn_node_embedding = x

            x = F.gumbel_softmax(x, tau=1, hard=True)

            self.gnn_node_concepts = x

            x = self.sum_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x*5)

            self.gnn_graph_local_concepts = x

            tab = self.mlp(x_tab)

            self.tab_activations = tab

            tab = self.batch_norm_img_1(tab)
            tab = F.sigmoid(tab*5)

            self.x_tab_local_concepts = tab

            d = 0

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = tab

            combined_concept = torch.cat((x, tab), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d
        elif prediction:
            combined_concept = torch.cat((x, x_tab), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, 0
        else:
            if mod1:
                x_tab = x_tab.reshape(batch[-1] + 1, self.num_in_features)
                tab = self.mlp(x_tab)
                tab = self.batch_norm_img_1(tab)
                tab = F.sigmoid(tab * 5)

                tab_aux = x.reshape(batch[-1] + 1, self.num_in_features)
                tab_aux = self.mlp(tab_aux)
                tab_aux = self.batch_norm_img_1(tab_aux)
                tab_aux = F.sigmoid(tab_aux * 5)

                return tab, tab_aux
            elif mod2:
                x = self.conv0(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv1(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv2(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv4(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv3(x, edge_index)
                x = F.leaky_relu(x)
                x = x.squeeze()
                x = F.gumbel_softmax(x, tau=1, hard=True)
                x = self.sum_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)

                x_g_aux = self.conv0(x_tab, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv1(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv2(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv4(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv3(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = x_g_aux.squeeze()
                x_g_aux = F.gumbel_softmax(x_g_aux, tau=1, hard=True)
                x_g_aux = self.sum_pool(x_g_aux, batch_aux)
                x_g_aux = self.batch_norm_g_1(x_g_aux)
                x_g_aux = F.sigmoid(x_g_aux * 5)

                return x, x_g_aux

class Tab_Graph_SingleCBM(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_SingleCBM, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )
        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 10),
                                  nn.ReLU(),
                                  nn.Linear(10, num_classes)
                                  )
        self.pred_t = nn.Sequential(nn.Linear(cluster_encoding_size, 10),
                                    nn.ReLU(),
                                    nn.Linear(10, num_classes)
                                    )

    def forward(self, x, edge_index, batch, x_tab, clean_features):

        x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)

        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = x.squeeze()

        x = F.gumbel_softmax(x, tau=1, hard=True)

        self.gnn_node_concepts = x

        x = self.sum_pool(x, batch)
        x = self.batch_norm_g_1(x)
        x = F.sigmoid(x * 5)

        self.gnn_graph_local_concepts = x

        out_g = self.pred_g(x)

        tab = self.mlp(x_tab)
        self.tab_activations = tab

        tab = self.batch_norm_img_1(tab)
        tab = F.sigmoid(tab*5)

        out_t = self.pred_t(tab)

        self.x_tab_local_concepts = tab

        self.gnn_graph_shared_concepts = x
        self.tab_shared_concepts = tab

        return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out_g, out_t

class Tab_Graph_Vanilla(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_Vanilla, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()


        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def forward(self, x, edge_index, batch, x_tab, clean_features):

        x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)

        x = self.conv0(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv2(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv4(x, edge_index)
        x = F.leaky_relu(x)

        x = self.conv3(x, edge_index)
        x = F.leaky_relu(x)

        x = x.squeeze()

        x = self.sum_pool(x, batch)

        self.gnn_graph_local_concepts = x


        tab = self.mlp(x_tab)

        self.x_tab_local_concepts = tab

        self.gnn_graph_shared_concepts = x
        self.tab_shared_concepts = tab

        combined_concept = torch.cat((x, tab), dim=-1)

        out = self.pred(combined_concept)
        d = 0

        return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d

class Tab_Graph_Vanilla_single(nn.Module):
    def __init__(self, num_in_features_gnn, num_in_features,
                 num_hidden_features_gnn, num_hidden_features,
                 cluster_encoding_size, num_classes):
        super(Tab_Graph_Vanilla_single, self).__init__()

        self.num_in_features = num_in_features

        self.conv0 = GCNConv(num_in_features_gnn, num_hidden_features_gnn)
        self.conv1 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv2 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv4 = GCNConv(num_hidden_features_gnn, num_hidden_features_gnn)
        self.conv3 = GCNConv(num_hidden_features_gnn, cluster_encoding_size)

        self.sum_pool = model_utils.SumPool()


        # linear layers
        self.mlp = nn.Sequential(nn.Linear(num_in_features, num_hidden_features),
                                 nn.ReLU(),
                                 nn.Linear(num_hidden_features, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )
        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 10),
                                  nn.ReLU(),
                                  nn.Linear(10, num_classes)
                                  )
        self.pred_t = nn.Sequential(nn.Linear(cluster_encoding_size, 10),
                                    nn.ReLU(),
                                    nn.Linear(10, num_classes)
                                    )

    def forward(self, x, edge_index, batch, x_tab, clean_features,
                missing=False, mod1=False, mod2=False, aux_edge=None, batch_aux=None):
        if not missing:
            x_tab = x_tab.reshape(batch[-1]+1, self.num_in_features)

            x = self.conv0(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv1(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv2(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv4(x, edge_index)
            x = F.leaky_relu(x)

            x = self.conv3(x, edge_index)
            x = F.leaky_relu(x)

            x = x.squeeze()

            x = self.sum_pool(x, batch)

            self.gnn_graph_local_concepts = x

            out_g = self.pred_g(x)

            tab = self.mlp(x_tab)

            self.x_tab_local_concepts = tab

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = tab

            out_t = self.pred_t(tab)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out_g, out_t
        else:
            if mod1:
                x_tab = x_tab.reshape(batch[-1] + 1, self.num_in_features)
                tab = self.mlp(x_tab)

                tab_aux = x.reshape(batch[-1] + 1, self.num_in_features)
                tab_aux = self.mlp(tab_aux)

                return tab, tab_aux
            elif mod2:
                x = self.conv0(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv1(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv2(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv4(x, edge_index)
                x = F.leaky_relu(x)
                x = self.conv3(x, edge_index)
                x = F.leaky_relu(x)
                x = x.squeeze()
                x = self.sum_pool(x, batch)

                x_g_aux = self.conv0(x_tab, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv1(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv2(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv4(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = self.conv3(x_g_aux, aux_edge)
                x_g_aux = F.leaky_relu(x_g_aux)
                x_g_aux = x_g_aux.squeeze()
                x_g_aux = self.sum_pool(x_g_aux, batch_aux)

                return x, x_g_aux

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


transform = T.Cartesian(cat=False)


class MNISTSuperpixels(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward_aux(self, x_image):

        x_image = self.conv(x_image)
        tmp = self.training
        if tmp:
            self.eval()
        x_image = self.batch_norm_img_1(x_image)
        if tmp:
            self.train()
        x_image_concepts = F.sigmoid(x_image)

        x_image = self.projection_tab(x_image_concepts)

        return x_image

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        x = self.batch_norm_g_1(x)
        x = F.sigmoid(x * 2)
        self.gnn_graph_local_concepts = x
        self.gnn_graph_activations = None
        out_g = self.pred_g(x)
        x = self.projection_graph(x)

        x_image = self.conv(x_image)
        self.image_activations = x_image
        x_image = self.batch_norm_img_1(x_image)
        x_image = F.sigmoid(x_image * 2)
        self.x_image_local_concepts = x_image
        out_img = self.pred_img(x_image)

        x_image = self.projection_tab(x_image)

        concat = torch.cat((x, x_image), dim=0)
        normalised = self.batch_norm_img_shared(concat)
        norm_concepts = F.sigmoid(normalised * 2)
        x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
        x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        d = self.dist(x, x_image, y_g, y_img)
        if math.isnan(d):
            d = 0

        combined_concept = torch.cat((x, x_image), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d

class MNISTSuperpixels_no_batch(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_no_batch, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward_aux(self, x_image):

        x_image = self.conv(x_image)
        tmp = self.training
        if tmp:
            self.eval()
        x_image = self.batch_norm_img_1(x_image)
        if tmp:
            self.train()
        x_image_concepts = F.sigmoid(x_image)

        x_image = self.projection_tab(x_image_concepts)

        return x_image

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data.batch = data.x_batch
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            # x = self.batch_norm_g_1(x)
            x = F.sigmoid(x)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)
            x = self.projection_graph(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            # x_image = self.batch_norm_img_1(x_image)
            x_image = F.sigmoid(x_image)
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            x_image = self.projection_tab(x_image)

            concat = torch.cat((x, x_image), dim=0)
            # normalised = self.batch_norm_img_shared(concat)
            norm_concepts = F.sigmoid(concat)
            x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
            x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            d = self.dist(x, x_image, y_g, y_img)
            if math.isnan(d):
                d = 0

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d
        elif prediction:
            combined_concept = torch.cat((data, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                aux_image = self.conv(aux_image)
                # aux_image = self.batch_norm_img_1(aux_image)
                aux_image = F.sigmoid(aux_image)
                aux_image = self.projection_tab(aux_image)

                x_image = self.conv(x_image)
                # x_image = self.batch_norm_img_1(x_image)
                x_image = F.sigmoid(x_image)
                x_image = self.projection_tab(x_image)

                concat = torch.cat((aux_image, x_image), dim=0)
                # normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(concat)
                aux_image = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                x_image.x = F.gumbel_softmax(x_image.x, tau=1, hard=True)
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)
                # x_image = self.batch_norm_g_1(x_image)
                x_image = F.sigmoid(x_image)
                x_image = self.projection_graph(x_image)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)
                # x = self.batch_norm_g_1(x)
                x = F.sigmoid(x)
                x = self.projection_graph(x)

                concat = torch.cat((x, x_image), dim=0)
                # normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(concat)
                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return x, x_image

class MNISTSuperpixels_missing(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_missing, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward_aux(self, x_image):

        x_image = self.conv(x_image)
        tmp = self.training
        if tmp:
            self.eval()
        x_image = self.batch_norm_img_1(x_image)
        if tmp:
            self.train()
        x_image_concepts = F.sigmoid(x_image)

        x_image = self.projection_tab(x_image_concepts)

        return x_image

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data.batch = data.x_batch
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 2)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)
            x = self.projection_graph(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            x_image = self.batch_norm_img_1(x_image)
            x_image = F.sigmoid(x_image * 2)
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            x_image = self.projection_tab(x_image)

            concat = torch.cat((x, x_image), dim=0)
            normalised = self.batch_norm_img_shared(concat)
            norm_concepts = F.sigmoid(normalised * 2)
            x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
            x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            d = self.dist(x, x_image, y_g, y_img)
            if math.isnan(d):
                d = 0

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d
        elif prediction:
            combined_concept = torch.cat((data, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                aux_image = self.conv(aux_image)
                aux_image = self.batch_norm_img_1(aux_image)
                aux_image = F.sigmoid(aux_image * 2)
                aux_image = self.projection_tab(aux_image)

                x_image = self.conv(x_image)
                x_image = self.batch_norm_img_1(x_image)
                x_image = F.sigmoid(x_image * 2)
                x_image = self.projection_tab(x_image)

                concat = torch.cat((aux_image, x_image), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 2)
                aux_image = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                x_image.x = F.gumbel_softmax(x_image.x, tau=1, hard=True)
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)
                x_image = self.batch_norm_g_1(x_image)
                x_image = F.sigmoid(x_image * 2)
                x_image = self.projection_graph(x_image)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 2)
                x = self.projection_graph(x)

                concat = torch.cat((x, x_image), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 2)
                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return x, x_image

class MNISTSuperpixels_SingleCBM(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_SingleCBM, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, local_num_classes)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, local_num_classes)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(local_num_classes, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, global_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(local_num_classes, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, global_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data.batch = data.x_batch
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        x = self.projection_graph(x)
        c_g = F.sigmoid(x)
        self.gnn_graph_local_concepts = c_g
        self.gnn_graph_activations = None
        out_g = self.pred_g(c_g)

        x_image = self.conv(x_image)
        x_image = self.projection_tab(x_image)
        self.image_activations = x_image
        c_image = F.sigmoid(x_image)
        self.x_image_local_concepts = c_image
        out_img = self.pred_img(c_image)

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out_g, out_img, c_g, c_image

class MNISTSuperpixels_MultiCBM(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_MultiCBM, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data.batch = data.x_batch
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 2)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            x_image = self.batch_norm_img_1(x_image)
            x_image = F.sigmoid(x_image * 2)
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, 0
        elif prediction:
            combined_concept = torch.cat((data, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                aux_image = self.conv(aux_image)
                aux_image = self.batch_norm_img_1(aux_image)
                aux_image = F.sigmoid(aux_image * 2)

                x_image = self.conv(x_image)
                x_image = self.batch_norm_img_1(x_image)
                x_image = F.sigmoid(x_image * 2)

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                x_image.x = F.gumbel_softmax(x_image.x, tau=1, hard=True)
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)
                x_image = self.batch_norm_g_1(x_image)
                x_image = F.sigmoid(x_image * 2)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 2)

            return x, x_image
class MNISTSuperpixels_Vanilla(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_Vanilla, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        self.gnn_graph_local_concepts = x
        self.gnn_graph_activations = None

        x_image = self.conv(x_image)
        self.image_activations = x_image
        self.x_image_local_concepts = x_image

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        d = 0

        combined_concept = torch.cat((x, x_image), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, None, None, d

class MNISTSuperpixels_SingleVanilla(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(MNISTSuperpixels_SingleVanilla, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 128),
                                  nn.ReLU(),
                                  nn.Linear(128, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, global_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, global_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None

            out_g = self.pred_g(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            self.x_image_local_concepts = x_image

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            out_img = self.pred_img(x_image)

            d = 0

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, None, out_g, out_img, d
        else:
            if mod1:
                aux_image = self.conv(aux_image)

                x_image = self.conv(x_image)

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)

                return x, x_image
class HalfMnist(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img, up_down):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and up_down[i] != up_down[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward_aux(self, x_image):

        x_image = self.conv(x_image)
        tmp = self.training
        if tmp:
            self.eval()
        x_image = self.batch_norm_img_1(x_image)
        if tmp:
            self.train()
        x_image_concepts = F.sigmoid(x_image)

        x_image = self.projection_tab(x_image_concepts)

        return x_image

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        up_down = data.up_down
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        x = self.batch_norm_g_1(x)
        x = F.sigmoid(x * 5)
        self.gnn_graph_local_concepts = x
        self.gnn_graph_activations = None
        out_g = self.pred_g(x)
        x = self.projection_graph(x)

        x_image = self.conv(x_image)
        self.image_activations = x_image
        x_image = self.batch_norm_img_1(x_image)
        x_image = F.sigmoid(x_image * 2)
        self.x_image_local_concepts = x_image
        out_img = self.pred_img(x_image)

        x_image = self.projection_tab(x_image)

        concat = torch.cat((x, x_image), dim=0)
        normalised = self.batch_norm_img_shared(concat)
        norm_concepts = F.sigmoid(normalised * 2)
        x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
        x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        d = self.dist(x, x_image, y_g, y_img, up_down)
        if math.isnan(d):
            d = 0

        combined_concept = torch.cat((x, x_image), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d

class HalfMnist_vanilla(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist_vanilla, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img, up_down):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and up_down[i] != up_down[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        up_down = data.up_down
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        self.gnn_graph_local_concepts = x
        self.gnn_graph_activations = None

        x_image = self.conv(x_image)
        self.image_activations = x_image
        self.x_image_local_concepts = x_image

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        d = 0

        combined_concept = torch.cat((x, x_image), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, None, None, d

class HalfMnist_vanilla_single(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist_vanilla_single, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img, up_down):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and up_down[i] != up_down[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            up_down = data.up_down
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            d = 0

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d
        else:
            if mod1:
                aux_image = self.conv(aux_image)

                x_image = self.conv(x_image)

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)

                return x, x_image
class HalfMnist_cbm_single(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist_cbm_single, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, local_num_classes)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, local_num_classes)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size * 2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(local_num_classes, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(local_num_classes, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img):
        up_down = data.up_down
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_add_pool(x, batch)
        x = self.projection_graph(x)
        c_g = F.sigmoid(x)
        self.gnn_graph_local_concepts = c_g
        self.gnn_graph_activations = None
        out_g = self.pred_g(c_g)

        x_image = self.conv(x_image)
        x_image = self.projection_tab(x_image)
        self.image_activations = x_image
        c_image = F.sigmoid(x_image)
        self.x_image_local_concepts = c_image
        out_img = self.pred_img(c_image)

        self.gnn_graph_shared_concepts = x
        self.x_image_shared_concepts = x_image

        return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out_g, out_img, c_g, c_image

class HalfMnist_MultiCBM(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist_MultiCBM, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            up_down = data.up_down
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 5)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            x_image = self.batch_norm_img_1(x_image)
            x_image = F.sigmoid(x_image * 2)
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            d = 0

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d
        elif prediction:
            combined_concept = torch.cat((data, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                aux_image = self.conv(aux_image)
                aux_image = self.batch_norm_img_1(aux_image)
                aux_image = F.sigmoid(aux_image * 2)

                x_image = self.conv(x_image)
                x_image = self.batch_norm_img_1(x_image)
                x_image = F.sigmoid(x_image * 2)

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                x_image.x = F.gumbel_softmax(x_image.x, tau=1, hard=True)
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)
                x_image = self.batch_norm_g_1(x_image)
                x_image = F.sigmoid(x_image * 2)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 2)

                return x, x_image

class HalfMnist_missing(torch.nn.Module):
    def __init__(self, cluster_encoding_size, local_num_classes, global_num_classes):
        super(HalfMnist_missing, self).__init__()
        self.conv1 = SplineConv(1, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, cluster_encoding_size, dim=2, kernel_size=5)
        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 7 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, cluster_encoding_size)
        )

        self.projection_graph = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, cluster_encoding_size)
                                              )
        #
        self.projection_tab = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, cluster_encoding_size)
                                            )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 32),
                                  nn.ReLU(),
                                  nn.Dropout(p=0.3),
                                  nn.Linear(32, global_num_classes)
                                  )

        self.pred_g = nn.Sequential(nn.Linear(cluster_encoding_size, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, local_num_classes)
                                    )

        self.pred_img = nn.Sequential(nn.Linear(12, 64),
                                      nn.ReLU(),
                                      nn.Linear(64, local_num_classes)
                                      )

    def dist(self, x_graph, x_image, y_g, y_img, up_down):
        d = nn.PairwiseDistance(p=2)

        img_indexes = []
        graph_indexes = []
        for i in range(x_image.shape[0]):
            if random.random() < 0.10:
                for j, g in enumerate(y_g):
                    if (y_img[i] == y_g[j] and up_down[i] != up_down[j] and j not in graph_indexes):
                        graph_indexes += [j]
                        img_indexes += [i]
                        break

        img_indexes = torch.Tensor(img_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_img_anchors = x_image[img_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_img_anchors, x_graph_anchors).mean()

    '''
    def dist(self, x_graph, aux_image):
        d = nn.PairwiseDistance(p=2)

        return d(aux_image, x_graph).mean()
    '''

    def forward_aux(self, x_image):

        x_image = self.conv(x_image)
        tmp = self.training
        if tmp:
            self.eval()
        x_image = self.batch_norm_img_1(x_image)
        if tmp:
            self.train()
        x_image_concepts = F.sigmoid(x_image)

        x_image = self.projection_tab(x_image_concepts)

        return x_image

    def forward(self, data, x_image, aux_image, anchors, y_g, y_img,
                missing=False, mod1=False, mod2=False, prediction=False):
        if not missing:
            up_down = data.up_down
            data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            data.edge_attr = None
            data = max_pool(cluster, data, transform=transform)

            data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
            data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
            weight = normalized_cut_2d(data.edge_index, data.pos)
            cluster = graclus(data.edge_index, weight, data.x.size(0))
            x, batch = max_pool_x(cluster, data.x, data.batch)

            x = global_add_pool(x, batch)
            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 5)
            self.gnn_graph_local_concepts = x
            self.gnn_graph_activations = None
            out_g = self.pred_g(x)
            x = self.projection_graph(x)

            x_image = self.conv(x_image)
            self.image_activations = x_image
            x_image = self.batch_norm_img_1(x_image)
            x_image = F.sigmoid(x_image * 2)
            self.x_image_local_concepts = x_image
            out_img = self.pred_img(x_image)

            x_image = self.projection_tab(x_image)

            concat = torch.cat((x, x_image), dim=0)
            normalised = self.batch_norm_img_shared(concat)
            norm_concepts = F.sigmoid(normalised * 2)
            x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
            x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

            self.gnn_graph_shared_concepts = x
            self.x_image_shared_concepts = x_image

            d = self.dist(x, x_image, y_g, y_img, up_down)
            if math.isnan(d):
                d = 0

            combined_concept = torch.cat((x, x_image), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.x_image_shared_concepts, out, out_g, out_img, d
        elif prediction:
            combined_concept = torch.cat((data, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                aux_image = self.conv(aux_image)
                aux_image = self.batch_norm_img_1(aux_image)
                aux_image = F.sigmoid(aux_image * 2)
                aux_image = self.projection_tab(aux_image)

                x_image = self.conv(x_image)
                x_image = self.batch_norm_img_1(x_image)
                x_image = F.sigmoid(x_image * 2)
                x_image = self.projection_tab(x_image)

                concat = torch.cat((aux_image, x_image), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 2)
                aux_image = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return aux_image, x_image
            elif mod2:
                x_image.x = F.elu(self.conv1(x_image.x, x_image.edge_index, x_image.edge_attr))
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image.edge_attr = None
                x_image = max_pool(cluster, x_image, transform=transform)

                x_image.x = F.elu(self.conv2(x_image.x, x_image.edge_index, x_image.edge_attr))
                x_image.x = F.gumbel_softmax(x_image.x, tau=1, hard=True)
                weight = normalized_cut_2d(x_image.edge_index, x_image.pos)
                cluster = graclus(x_image.edge_index, weight, x_image.x.size(0))
                x_image, batch = max_pool_x(cluster, x_image.x, x_image.batch)

                x_image = global_add_pool(x_image, batch)
                x_image = self.batch_norm_g_1(x_image)
                x_image = F.sigmoid(x_image * 2)
                x_image = self.projection_graph(x_image)

                data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                data.edge_attr = None
                data.batch = data.x_batch
                data = max_pool(cluster, data, transform=transform)

                data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
                data.x = F.gumbel_softmax(data.x, tau=1, hard=True)
                weight = normalized_cut_2d(data.edge_index, data.pos)
                cluster = graclus(data.edge_index, weight, data.x.size(0))
                x, batch = max_pool_x(cluster, data.x, data.batch)

                x = global_add_pool(x, batch)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 2)
                x = self.projection_graph(x)

                concat = torch.cat((x, x_image), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 2)
                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_image = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return x, x_image

class CLEVR_SHARCS(nn.Module):
    def __init__(self, num_in_features,
                 cluster_encoding_size, num_classes):
        super(CLEVR_SHARCS, self).__init__()

        self.num_in_features = num_in_features
        self.epoch = 0
        self.threshold = 30
        self.mode = 'sequential'

        self.text_encoder = nn.Sequential(nn.Linear(num_in_features, cluster_encoding_size*2),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size*2, cluster_encoding_size)
                                 )

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.conv = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer
        for param in self.conv.fc.parameters():
            param.requires_grad = True
        self.conv_linear = nn.Linear(1000, cluster_encoding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.projection_text = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_image = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, y):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if y[i] == 1:
                if random.random() < 0.20:
                    graph_indexes += [i]
                    tab_indexes += [i]

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x_text, x_image, y, missing=False, prediction=False, mod1=False, mod2=False):
        if not missing:
            x = self.text_encoder(x_text)
            x = F.leaky_relu(x)

            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 5)
            self.gnn_graph_local_concepts = x


            x_2 = self.conv(x_image)
            x_2 = self.dropout(x_2)
            x_2 = self.conv_linear(x_2)
            x_2 = F.leaky_relu(x_2)

            x_2 = self.batch_norm_img_1(x_2)
            x_2 = F.sigmoid(x_2 * 5)
            self.x_tab_local_concepts = x_2


            if self.epoch > self.threshold or self.mode != 'sequential':
                x = self.projection_text(x)
                x_2 = self.projection_image(x_2)
                concat = torch.cat((x, x_2), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised*5)

                x = norm_concepts[:int(norm_concepts.shape[0]/2)]
                x_2 = norm_concepts[int(norm_concepts.shape[0]/2):]

                d = self.dist(x_2, x, y)
                if math.isnan(d):
                  d = 0
            else:
                d = 0

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = x_2

            combined_concept = torch.cat((x, x_2), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d
        elif prediction:
            combined_concept = torch.cat((x_text, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                x = self.text_encoder(x_text)
                x = F.leaky_relu(x)

                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)

                x_aux = self.text_encoder(x_image)
                x_aux = F.leaky_relu(x_aux)

                x_aux = self.batch_norm_g_1(x_aux)
                x_aux = F.sigmoid(x_aux * 5)

                x = self.projection_text(x)
                x_aux = self.projection_text(x_aux)
                concat = torch.cat((x, x_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return x, x_aux
            elif mod2:
                x = self.conv(x_image)
                x = self.dropout(x)
                x = self.conv_linear(x)
                x = F.leaky_relu(x)

                x = self.batch_norm_img_1(x)
                x = F.sigmoid(x * 5)

                x_aux = self.conv(x_text)
                x_aux = self.dropout(x_aux)
                x_aux = self.conv_linear(x_aux)
                x_aux = F.leaky_relu(x_aux)

                x_aux = self.batch_norm_img_1(x_aux)
                x_aux = F.sigmoid(x_aux * 5)

                x = self.projection_image(x)
                x_aux = self.projection_image(x_aux)
                concat = torch.cat((x, x_aux), dim=0)
                normalised = self.batch_norm_img_shared(concat)
                norm_concepts = F.sigmoid(normalised * 5)

                x = norm_concepts[:int(norm_concepts.shape[0] / 2)]
                x_aux = norm_concepts[int(norm_concepts.shape[0] / 2):]

                return x, x_aux



class CLEVR_MultiCBM(nn.Module):
    def __init__(self, num_in_features,
                 cluster_encoding_size, num_classes):
        super(CLEVR_MultiCBM, self).__init__()

        self.num_in_features = num_in_features

        self.text_encoder = nn.Sequential(nn.Linear(num_in_features, cluster_encoding_size*2),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size*2, cluster_encoding_size)
                                 )

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.conv = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer
        for param in self.conv.fc.parameters():
            param.requires_grad = True
        self.conv_linear = nn.Linear(1000, cluster_encoding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.projection_text = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_image = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, y):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if y[i] == 1:
                if random.random() < 0.20:
                    graph_indexes += [i]
                    tab_indexes += [i]

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x_text, x_image, y, missing=False, prediction=False, mod1=False, mod2=False):
        if not missing:
            x = self.text_encoder(x_text)
            x = F.leaky_relu(x)

            x = self.batch_norm_g_1(x)
            x = F.sigmoid(x * 5)
            self.gnn_graph_local_concepts = x

            x_2 = self.conv(x_image)
            x_2 = self.dropout(x_2)
            x_2 = self.conv_linear(x_2)
            x_2 = F.leaky_relu(x_2)

            x_2 = self.batch_norm_img_1(x_2)
            x_2 = F.sigmoid(x_2 * 5)
            self.x_tab_local_concepts = x_2

            d = 0

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = x_2

            combined_concept = torch.cat((x, x_2), dim=-1)

            out = self.pred(combined_concept)

            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d
        elif prediction:
            combined_concept = torch.cat((x_text, x_image), dim=-1)

            out = self.pred(combined_concept)

            return out
        else:
            if mod1:
                x = self.text_encoder(x_text)
                x = F.leaky_relu(x)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)

                x_aux = self.text_encoder(x_image)
                x_aux = F.leaky_relu(x_aux)
                x_aux = self.batch_norm_g_1(x_aux)
                x_aux = F.sigmoid(x_aux * 5)

                return x, x_aux
            elif mod2:
                x = self.conv(x_image)
                x = self.dropout(x)
                x = self.conv_linear(x)
                x = F.leaky_relu(x)
                x = self.batch_norm_g_1(x)
                x = F.sigmoid(x * 5)

                x_aux = self.conv(x_text)
                x_aux = self.dropout(x_aux)
                x_aux = self.conv_linear(x_aux)
                x_aux = F.leaky_relu(x_aux)
                x_aux = self.batch_norm_g_1(x_aux)
                x_aux = F.sigmoid(x_aux * 5)

                return x, x_aux

class CLEVR_SingleCBM(nn.Module):
    def __init__(self, num_in_features,
                 cluster_encoding_size, num_classes):
        super(CLEVR_SingleCBM, self).__init__()

        self.num_in_features = num_in_features

        self.text_encoder = nn.Sequential(nn.Linear(num_in_features, cluster_encoding_size*2),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size*2, cluster_encoding_size)
                                 )

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.conv = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer
        for param in self.conv.fc.parameters():
            param.requires_grad = True
        self.conv_linear = nn.Linear(1000, cluster_encoding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.projection_text = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, num_classes)
                                 )

        self.projection_image = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, num_classes)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, y):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if y[i] == 1:
                if random.random() < 0.20:
                    graph_indexes += [i]
                    tab_indexes += [i]

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x_text, x_image, y):

        x = self.text_encoder(x_text)
        x = F.leaky_relu(x)

        x = self.batch_norm_g_1(x)
        x = F.sigmoid(x * 5)
        self.gnn_graph_local_concepts = x
        out_1 = self.projection_text(x)

        x_2 = self.conv(x_image)
        x_2 = self.dropout(x_2)
        x_2 = self.conv_linear(x_2)
        x_2 = F.leaky_relu(x_2)

        x_2 = self.batch_norm_img_1(x_2)
        x_2 = F.sigmoid(x_2 * 5)
        self.x_tab_local_concepts = x_2
        out_2 = self.projection_image(x_2)

        self.gnn_graph_shared_concepts = x
        self.tab_shared_concepts = x_2


        return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out_1, out_2

class CLEVR_SingleVanilla(nn.Module):
    def __init__(self, num_in_features,
                 cluster_encoding_size, num_classes):
        super(CLEVR_SingleVanilla, self).__init__()

        self.num_in_features = num_in_features

        self.text_encoder = nn.Sequential(nn.Linear(num_in_features, cluster_encoding_size*2),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size*2, cluster_encoding_size)
                                 )

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.conv = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer
        for param in self.conv.fc.parameters():
            param.requires_grad = True
        self.conv_linear = nn.Linear(1000, cluster_encoding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.projection_text = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, num_classes)
                                 )

        self.projection_image = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, num_classes)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, y):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if y[i] == 1:
                if random.random() < 0.20:
                    graph_indexes += [i]
                    tab_indexes += [i]

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x_text, x_image, y, missing=False, prediction=False, mod1=False, mod2=False):
        if not missing:
            x = self.text_encoder(x_text)
            x = F.leaky_relu(x)

            self.gnn_graph_local_concepts = x
            out_1 = self.projection_text(x)

            x_2 = self.conv(x_image)
            x_2 = self.dropout(x_2)
            x_2 = self.conv_linear(x_2)
            x_2 = F.leaky_relu(x_2)

            self.x_tab_local_concepts = x_2
            out_2 = self.projection_image(x_2)

            self.gnn_graph_shared_concepts = x
            self.tab_shared_concepts = x_2


            return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out_1, out_2
        else:
            if mod1:
                x = self.text_encoder(x_text)
                x = F.leaky_relu(x)

                x_aux = self.text_encoder(x_image)
                x_aux = F.leaky_relu(x_aux)

                return x, x_aux
            elif mod2:
                x = self.conv(x_image)
                x = self.dropout(x)
                x = self.conv_linear(x)
                x = F.leaky_relu(x)

                x_aux = self.conv(x_text)
                x_aux = self.dropout(x_aux)
                x_aux = self.conv_linear(x_aux)
                x_aux = F.leaky_relu(x_aux)

                return x, x_aux



class CLEVR_Vanilla(nn.Module):
    def __init__(self, num_in_features,
                 cluster_encoding_size, num_classes):
        super(CLEVR_Vanilla, self).__init__()

        self.num_in_features = num_in_features

        self.text_encoder = nn.Sequential(nn.Linear(num_in_features, cluster_encoding_size*2),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size*2, cluster_encoding_size)
                                 )

        self.sum_pool = model_utils.SumPool()

        self.batch_norm_g_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_1 = nn.BatchNorm1d(cluster_encoding_size, affine=False)
        self.batch_norm_img_shared = nn.BatchNorm1d(cluster_encoding_size, affine=False)

        # linear layers
        self.conv = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        for param in self.conv.parameters():
            param.requires_grad = False

        # Unfreeze the last fully-connected layer
        for param in self.conv.fc.parameters():
            param.requires_grad = True
        self.conv_linear = nn.Linear(1000, cluster_encoding_size)
        self.dropout = nn.Dropout(p=0.5)

        self.projection_text = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.projection_image = nn.Sequential(nn.Linear(cluster_encoding_size, cluster_encoding_size),
                                 nn.ReLU(),
                                 nn.Linear(cluster_encoding_size, cluster_encoding_size)
                                 )

        self.pred = nn.Sequential(nn.Linear(cluster_encoding_size*2, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, num_classes)
                                 )

    def dist(self, x_tab, x_graph, y):
        d = nn.PairwiseDistance(p=2)

        tab_indexes = []
        graph_indexes = []
        for i in range(x_tab.shape[0]):
            if y[i] == 1:
                if random.random() < 0.20:
                    graph_indexes += [i]
                    tab_indexes += [i]

        tab_indexes = torch.Tensor(tab_indexes).long()
        graph_indexes = torch.Tensor(graph_indexes).long()

        x_tab_anchors = x_tab[tab_indexes]
        x_graph_anchors = x_graph[graph_indexes]

        return d(x_tab_anchors, x_graph_anchors).mean()

    def forward(self, x_text, x_image, y):

        x = self.text_encoder(x_text)
        x = F.leaky_relu(x)

        self.gnn_graph_local_concepts = x

        x_2 = self.conv(x_image)
        x_2 = self.dropout(x_2)
        x_2 = self.conv_linear(x_2)
        x_2 = F.leaky_relu(x_2)

        self.x_tab_local_concepts = x_2

        d = 0

        self.gnn_graph_shared_concepts = x
        self.tab_shared_concepts = x_2

        combined_concept = torch.cat((x, x_2), dim=-1)

        out = self.pred(combined_concept)

        return self.gnn_graph_shared_concepts, self.tab_shared_concepts, out, d

class PredModel(nn.Module):
    def __init__(self, num_in_features,
                 hidden_dim, num_classes):
        super(PredModel, self).__init__()

        self.pred = nn.Sequential(nn.Linear(num_in_features, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, num_classes)
                                             )

    def forward(self, mod1, mod2):
        combined = torch.cat((mod1, mod2), dim=-1)

        out = self.pred(combined)

        return out

class ActivationClassifierConcepts():
    def __init__(self, y, concepts, train_mask, test_mask, index=True, max_depth=None):
        self.concepts_index = concepts
        self.max_depth = max_depth
        self.y = y
        self.index = index
        self.train_mask = train_mask
        self.test_mask = test_mask

        self.classifier, self.accuracy = self._train_classifier()

    def _train_classifier(self):
        self.train_concepts = []
        self.test_concepts = []

        for node_idx in range(len(self.train_mask)):
            if self.train_mask[node_idx] == 1:
                if self.index:
                    self.train_concepts.append([self.activation_to_concept(node_idx)])
                else:
                    self.train_concepts.append(self.activation_to_concept(node_idx))
            else:
                if self.index:
                    self.test_concepts.append([self.activation_to_concept(node_idx)])
                else:
                    self.test_concepts.append(self.activation_to_concept(node_idx))

        self.concepts = self.train_concepts + self.test_concepts

        cls = tree.DecisionTreeClassifier(max_depth=self.max_depth)
        cls = cls.fit(self.train_concepts, self.y[self.train_mask])

        # decision tree accuracy
        accuracy = cls.score(self.test_concepts, self.y[self.test_mask])

        return cls, accuracy

    def get_classifier_accuracy(self):
        return self.accuracy

    def activation_to_concept(self, node):
        # return cluster number as substitute of concept
        concept = self.concepts_index[node]

        return concept

    def concept_to_class(self, concept):
        concept = concept.reshape(1, -1)
        pred = self.classifier.predict(concept)

        return pred

    def plot(self, path, layer_num=0, k=0, reduction_type=None):
        fig, ax = plt.subplots(figsize=(20, 20))
        tree.plot_tree(self.classifier, ax=ax, filled=True)
        fig.suptitle(f"Decision Tree for {reduction_type} Model")
        wandb.log({f'tree_structure': wandb.Image(plt)})
        plt.savefig(os.path.join(path, f"{k}k_{layer_num}layer_{reduction_type}.png"))
        plt.show()

    def plot2(self, path, integer=1, layers=[0, 10]):

        viz_model = tree_viz.model(self.classifier,
                                   X_train=np.array(self.concepts), y_train=self.y.detach().numpy(),
                                   feature_names = ['Concepts']
                                    )

        v = viz_model.view(path=path, depth_range_to_display=layers)  # render as SVG into internal object
        v.show()  # pop up window
        v.save(f"{path}/concepts_tree{integer}.svg")

        v = viz_model.view2(path=path, depth_range_to_display=layers)  # render as SVG into internal object
        v.show()  # pop up window
        v.save(f"{path}/concepts_tree{integer+1}.svg")