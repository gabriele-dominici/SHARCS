import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import seaborn as sns
from matplotlib import rc
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
import wandb
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

import clustering_utils
from sklearn.cluster import KMeans, MeanShift, DBSCAN, AgglomerativeClustering
import torch
import models


def set_rc_params():
    small = 14
    medium = 20
    large = 28

    plt.rc('figure', autolayout=True, figsize=(12, 8))
    plt.rc('font', size=medium)
    plt.rc('axes', titlesize=medium, labelsize=small, grid=True)
    plt.rc('xtick', labelsize=small)
    plt.rc('ytick', labelsize=small)
    plt.rc('legend', fontsize=small)
    plt.rc('figure', titlesize=large, facecolor='white')
    plt.rc('legend', loc='upper left')


def plot_model_accuracy(train_accuracies, test_accuracies, model_name, path):
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Testing Accuracy")
    plt.title(f"Accuracy of {model_name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_accuracy_plot.svg"))
    plt.savefig(os.path.join(path, f"model_accuracy_plot.png"))
    wandb.log({'accuracy': wandb.Image(plt)})
    plt.show()


def plot_model_loss(train_losses, test_losses, model_name, path):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Testing Loss")
    plt.title(f"Loss of {model_name} Model during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(path, f"model_loss_plot.pdf"))
    plt.savefig(os.path.join(path, f"model_loss_plot.png"))
    wandb.log({'loss': wandb.Image(plt)})
    plt.show()


# should be called with concepts
def plot_concept_heatmap(centroids, concepts, y, used_centroid_labels, model_name, layer_num, path, id_title="", id_path=""):
    LATEX_SYMBOL = ""  # Change to "$" if working out of server
    rc('text', usetex=(LATEX_SYMBOL == "$"))
    plt.style.use('seaborn-whitegrid')

    plt.figure(figsize=[15, 5])
    fig, ax = plt.subplots(len(np.unique(used_centroid_labels)), 2, gridspec_kw={'width_ratios': [5, 1]})
    # fig.set_size_inches(12, len(centroids) * 0.8)
    fig.suptitle(f"{id_title}Concept Heatmap of the {model_name} with concepts extracted from Layer {layer_num}")

    if torch.is_tensor(concepts):
        concepts = concepts.detach().numpy()
    else:
        concepts = concepts[:, np.newaxis]

    used_centroid_labels = used_centroid_labels.squeeze()

    nclasses = len(np.unique(y))
    if len(centroids) == 1:
        sns.heatmap(concepts[used_centroid_labels==0] > 0.5, cbar=None, ax=ax[0])
        sns.heatmap(y[used_centroid_labels==0].unsqueeze(-1), vmin=0, vmax=4, cmap="Set2", ax=ax[1])
    else:
        for i in range(len(np.unique(used_centroid_labels))):
            sns.heatmap(concepts[used_centroid_labels==i] > 0.5, cbar=None, ax=ax[i, 0],
                        xticklabels=False, yticklabels=False)
            sns.heatmap(y[used_centroid_labels==i].unsqueeze(-1), vmin=0, vmax=nclasses, cmap="Set2", ax=ax[i, 1],
                        xticklabels=False, yticklabels=False, cbar=None)

    legend_elements = [Patch(facecolor=c, label=f'Class {i // 2}') for i, c in
                       enumerate(sns.color_palette("Set2", 2 * nclasses)) if i % 2 == 0]
    fig.legend(handles=legend_elements, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(legend_elements))
    plt.savefig(os.path.join(path, f"{id_path}concept_heatmap.pdf"))
    plt.savefig(os.path.join(path, f"{id_path}concept_heatmap_layer.png"))
    wandb.log({id_path: wandb.Image(plt)})


def plot_clustering(seed, activation, y, centroids, centroid_labels, used_centroid_labels, model_name, layer_num, path, task="local", id_path="node",
                    extra=False, train_mask=None, test_mask=None, n_classes=10, all=True):

    all_data = np.vstack([activation, centroids])

    tsne_model = TSNE(n_components=2, random_state=seed)
    all_data2d = tsne_model.fit_transform(all_data)

    d = all_data2d[:len(activation)]
    centroids2d = all_data2d[len(activation):]

    fig = plt.figure(figsize=[15, 5])
    fig.suptitle(f"Clustering of {task} concepts")

    ax = plt.subplot(1, 3, 1)
    if task == 'shared':
        ax.set_title("Two modalities")
    else:
        ax.set_title("Real Labels")
    p = sns.color_palette("husl", len(np.unique(y)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=y, palette=p)

    ax = plt.subplot(1, 3, 2)
    ax.set_title("Model's Clusters")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=used_centroid_labels, palette=p, legend=None)

    ax = plt.subplot(1, 3, 3)
    ax.set_title("Model's Cluster Centroids")
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=used_centroid_labels, palette=p, legend=None, alpha=0.3)

    p = sns.color_palette("husl", len(centroids))
    sns.scatterplot(centroids2d[:, 0], centroids2d[:, 1], hue=list(range(len(centroids))), palette=p, alpha=0.7,
                    legend=None, **{'s': 600, 'marker': '*', 'edgecolors': None})

    plt.savefig(os.path.join(path, f"DifferentialClustering_Raw_Layer{task}.png"))
    plt.savefig(os.path.join(path, f"DifferentialClustering_Raw_Layer{task}.pdf"))
    wandb.log({f'distribution_{task}': wandb.Image(plt)})
    plt.show()


def plot_clustering_images_inside(seed, activation, centroids, images, used_centroid_labels, path, task="local", id_path="node"):
    all_data = np.vstack([activation, centroids])

    tsne_model = TSNE(n_components=2, random_state=seed)
    all_data2d = tsne_model.fit_transform(all_data)

    d = all_data2d[:len(activation)]
    centroids2d = all_data2d[len(activation):]

    fig = plt.figure(figsize=[20, 10])
    fig.suptitle(f"Model's Cluster Centroids")
    ax = plt.subplot(1, 1, 1)
    p = sns.color_palette("husl", len(np.unique(used_centroid_labels)))
    sns.scatterplot(d[:, 0], d[:, 1], hue=used_centroid_labels, palette=p, legend=None, alpha=0.3)

    for i in range(len(centroids2d)):
        imagebox = OffsetImage(images[i], zoom=0.15)
        ab = AnnotationBbox(imagebox, centroids2d[i], frameon=False)
        ax.add_artist(ab)
    wandb.log({f'distribution_{task}': wandb.Image(plt)})
    fig.tight_layout()
    plt.savefig(os.path.join(path, f"Shared_space.png"))
    plt.savefig(os.path.join(path, f"Shared_space.pdf"))
    plt.show()






def print_cluster_counts(used_centroid_labels):
    cluster_counts = np.unique(used_centroid_labels, return_counts=True)

    cluster_ids, counts = zip(*sorted(zip(cluster_counts[0], cluster_counts[1])))
    print("Cluster sizes by cluster id:")
    for id, c in zip(cluster_ids, counts):
        print(f"\tCluster {id}: {c}")

    return cluster_counts