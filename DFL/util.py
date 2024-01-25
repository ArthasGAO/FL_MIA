import copy
import random
import sys
from collections import Counter

from data_samples.CIFAR10 import CIFAR10Dataset
from data_samples.MNIST import MNISTDataset
from model.MLP_MNIST import MLP
from torch.utils.data import DataLoader, Dataset, random_split
from Node import Node

from trainer import torch


def dataset_model_set(dataset, model, logger):
    logger.info(f"This time dataset used is {dataset}!")
    g_dataset = None
    g_model = None
    if dataset == "Mnist":
        g_dataset = MNISTDataset()
        if model == "mlp":
            g_model = MLP(784, 1e-3, 10)
        elif model == "cnn":
            pass
        else:
            pass
    elif dataset == "FMnist":
        pass
    elif dataset == "Cifar10":
        g_dataset = CIFAR10Dataset(sub_id=0, number_sub=1, root_dir=f"{sys.path[0]}/data", iid=True)
        if model == "mlp":
            pass
        elif model == "cnn":
            pass
    return g_dataset, g_model


def distribute_data_equally(dataset: Dataset, num_clients: int, batch_size, sign, iid, logger):
    """
    Distributes a dataset equally among a specified number of clients.

    :param logger: which file we want to log information
    :param batch_size: size for each batch used in dataloader
    :param iid: this means if we mock the iid data distribution for each client
    :param sign: indicate shuffle or not
    :param dataset: The dataset to distribute.
    :param num_clients: The number of clients among which to distribute the data.
    :return: A list of DataLoaders, each representing the local dataset for a client.
    """
    # Determine the size of each partition
    logger.info(f"dataset used iid is {int(iid)}!")
    total_size = len(dataset)
    partition_size = total_size // num_clients
    remainder = total_size % num_clients

    # Create partitions for each client, with the remainder distributed to the first few
    partitions = [partition_size + (1 if i < remainder else 0) for i in range(num_clients)]

    total_labels = Counter(y for _, y in dataset)
    logger.info(f"Total dataset size: {total_size}")
    logger.info(f"Number of clients: {num_clients}")
    logger.info(f"Label distribution in the full dataset: {total_labels}")

    # Split the dataset into non-overlapping new datasets of equal size
    client_datasets = random_split(dataset, partitions)

    client_dataloaders = []
    for i, client_dataset in enumerate(client_datasets):
        client_loader = DataLoader(dataset=client_dataset, batch_size=batch_size, shuffle=sign, num_workers=12)
        client_dataloaders.append(client_loader)

        # Count labels in each client dataset
        label_counts = Counter()
        for _, label in client_dataset:
            label_counts[label] += 1

        logger.info(f"Client {i + 1} partition size: {len(client_dataset)}")
        logger.info(f"Client {i + 1} label distribution: {label_counts}")

    return client_dataloaders


def average_weights(local_weights):
    """
    Average the weights of the local models to update the global model.
    """
    global_weights = copy.deepcopy(local_weights[0])
    for key in global_weights.keys():
        for i in range(1, len(local_weights)):
            global_weights[key] += local_weights[i][key]
        global_weights[key] = torch.div(global_weights[key], len(local_weights))
    return global_weights


def parse_experiment_file(file_path):
    experiment_info = {}

    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into key and value parts
            parts = line.strip().split(':')
            if len(parts) == 2:
                key, value = parts[0].strip(), parts[1].strip()
                # Store the key-value pair in the dictionary
                experiment_info[key] = value

    return experiment_info


def create_nodes_list(num, train_dataloaders, test_dataloaders, val_dataloaders, model):
    node_list = []
    for i in range(1, num + 1):
        node = Node(i, copy.deepcopy(model), train_dataloaders[i - 1], val_dataloaders[i - 1], test_dataloaders[i - 1])
        node_list.append(node)
    return node_list


def create_adjacency(nodes_list, topology):
    num_nodes = len(nodes_list)
    random.shuffle(nodes_list)  # include randomness

    if topology == "fully":
        for node in nodes_list:
            node.neigh = {s for s in nodes_list if s != node}
    elif topology == "star":
        root = nodes_list[0]
        nodes_list[0].neigh = {s for s in nodes_list if s != root}
        for node in nodes_list:
            if node != root:
                node.neigh = {root}
    elif topology == "ring":
        for i, node in enumerate(nodes_list):
            left_neighbor = nodes_list[(i - 1) % num_nodes]
            right_neighbor = nodes_list[(i + 1) % num_nodes]
            node.neigh = {left_neighbor, right_neighbor}
    else:
        raise ValueError("Unsupported topology")

    return nodes_list
