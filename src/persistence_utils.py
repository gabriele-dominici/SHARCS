import joblib
import os
import torch

def save_config(seed, dataset_name, model_name, num_classes, k, train_test_split, num_hidden_units, epochs, lr, num_nodes_view, num_expansions, layer_num, layer_key, path):
    config = {'seed': seed,
               'dataset_name': dataset_name,
               'model_name': model_name,
               'num_classes': num_classes,
               'k': k,
               'train_test_split': train_test_split,
               'num_hidden_units': num_hidden_units,
               'epochs': epochs,
               'lr': lr,
               'num_nodes_view': num_nodes_view,
               'num_expansions': num_expansions,
               'layer_num': layer_num,
               'layer_key': layer_key
              }
    persist_experiment(config, path, 'config.z')

def persist_experiment(data, path, filename):
    filepath = os.path.join(path, filename)
    joblib.dump(data, filepath, compress=True)


def load_experiment(path, filename):
    filepath = os.path.join(path, filename)
    return joblib.load(filepath)


def persist_model(model, path, filename):
    filepath = os.path.join(path, filename)
    torch.save(model.state_dict(), filepath)


def load_model(model, path, filename):
    filepath = os.path.join(path, filename)
    model.load_state_dict(torch.load(filepath))
    model.eval()

    return model
