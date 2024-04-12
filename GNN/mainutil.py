# python general import
import os
import glob
import yaml
import datetime
import argparse

# ext array library
import awkward as ak
import numpy as np
from typing import Optional

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# torch and PyG
import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, NNConv, global_add_pool, global_mean_pool, MetaLayer, MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

# torch multiprocessing: DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# graph viewer
import networkx as nx
#from torch_scatter import scatter

# user import
#from util import obj, PlotUtil, count_model_params
#from buildDataset import ZtoTauTauDataset
#from buildModel import NuNet, get_model_kwargs
#from train import Trainer, TrainerDDP


# Load config
def load_config(filepath):
    print("Loading config")
    config = {}
    with open(filepath, 'r') as temp:
        config = yaml.safe_load(temp)
    return config

# Plotting Graph#
def plot_nx_graph(dataset, outdir):
    import random
    dsize = len(dataset)
    idxs = random.sample(list(range(dsize)), 25)

    # Create a canvas with a 10x10 grid of subplots
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))

    _idx = 0
    for i in range(5):
        for j in range(5):
            evt_idx = idxs[_idx]
            data=dataset[evt_idx].cpu()
            nxg = to_networkx(data)
            pos = {i: (data.x[i, 1], data.x[i, 2]) for i in nxg.nodes}
            ax  = axs[i,j]
            nx.draw_networkx(nxg, pos, with_labels=True, arrows=False, node_size=5*data.x[:, 0], node_shape="o", ax=ax)
            ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
            ax.set_title(f"event {evt_idx}")
            _idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "nx_graph.png"), dpi=300)


# For DDP: Each process control a single gpu
# https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
# https://medium.com/polo-club-of-data-science/multi-gpu-training-in-pytorch-with-code-part-3-distributed-data-parallel-d26e93f17c62
def ddp_setup(rank: int, world_size: int) -> None:
    """
    ags:
       rank:
       world_size:
    """
    os.environ["MASTER_ADDR"] = "localhost"
    # select any idle port on your machine
    os.environ["MASTER_PORT"] = "12355"

    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_setup_torchrun():
    init_process_group(backend="nccl")


# split dataset
def dataset_split(dataset: Dataset, frac: float):
    #dataset = dataset.shuffle()
    n_total = dataset.len()

    n_train = int(np.floor(n_total * frac))
    n_val   = int(np.floor((n_total - n_train) * 0.5)) 

    train_dataset = dataset[:n_train]
    val_dataset   = dataset[n_train:(n_train+n_val)]
    test_dataset  = dataset[(n_train+n_val):] 
    
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(val_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')

    return train_dataset, val_dataset, test_dataset


# prepare dataloader
def prepare_dataloader(dataset: Dataset, 
                       batch_size: int,
                       sampler_shuffle: bool,
                       ddp: bool):
    sampler = None

    loader = DataLoader(dataset, 
                        batch_size=batch_size,
                        shuffle=False)
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=sampler_shuffle)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            sampler=sampler)
                            #num_workers=4)
    return loader, sampler


def prepare_dataloaders(train_dataset: Dataset, 
                        val_dataset: Dataset, 
                        test_dataset: Dataset, 
                        batch_size: int, 
                        ddp: bool):
    
    train_loader, train_sampler = prepare_dataloader(train_dataset, batch_size, True, ddp)
    val_loader, _ = prepare_dataloader(val_dataset, batch_size, False, ddp)
    test_loader, _ = prepare_dataloader(test_dataset, batch_size, False, ddp)

    return train_loader, train_sampler, val_loader, test_loader


def plot_evals(trues, preds):
    pass


def normalise_column(column):
    mask = ((column.max() - column.min()) == 0) | (abs(column).sum() == 0) 
    if mask:  # Exclude the column with all zeros
        return column
    else:
        return (column - column.min()) / (column.max() - column.min())
