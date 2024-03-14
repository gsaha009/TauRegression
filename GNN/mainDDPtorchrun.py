# PythonnnTrainer = TrainerDDPtorchrun(gpu_id = gpu_id, general import
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
mp.set_start_method('spawn', force=True)
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# graph viewer
import networkx as nx
#from torch_scatter import scatter

# user import
from util import obj, PlotUtil, count_model_params
from buildDataset import ZtoTauTauDataset
from buildModel import NuNet, get_model_kwargs
from train import TrainerDDPtorchrun

from mainutil import *


# Main function
def main(configfile: str, doplot: bool):
    # initialize ddp
    ddp_setup_torchrun()
    local_rank = torch.distributed.get_rank()
    print(f"Rank: {local_rank}")
    #gpu_id = torch.device("cuda", local_rank)
    #print(f"device: {gpu_id}")
    gpu_id = local_rank
    print(f"GPU_ID: {gpu_id}")
    #gpu_id = int(os.environ["LOCAL_RANK"])

    datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config_dict = obj(load_config(configfile))

    torch.manual_seed(config_dict.torch_seed)

    input_path = config_dict.inputs.datapath
    node_file = f"{input_path}/{config_dict.inputs.node_file}"
    target_file = f"{input_path}/{config_dict.inputs.target_file}"
    global_file = f"{input_path}/{config_dict.inputs.global_file}"

    outputpath = config_dict.outputpath
    outdir = f"{outputpath}_{datetime_tag}"
    os.makedirs(outdir, exist_ok=True)

    ckptpath = f"{config_dict.modelpath}_{datetime_tag}"
    os.makedirs(ckptpath, exist_ok=True)
    
    dotrain = config_dict.train.cmd
    doval = config_dict.val.cmd
    dotest = config_dict.test.cmd
    final_model_path = config_dict.test.final_model_path
    doeval = config_dict.evaluate.cmd

    node_info = ak.from_parquet(node_file)
    target_info = ak.from_parquet(target_file)
    global_info = ak.from_parquet(global_file)

    events_record = {}
    events_record["node_feats"] = node_info
    events_record["target_feats"] = target_info
    events_record["global_feats"] = global_info

    
    if doplot:
        print("Plotting")
        plotter = PlotUtil(events_record, outdir)
        plotter.plot_nodes()
        plotter.plot_targets()
        plotter.plot_globals()

    print("Preparing PyG Dataset")
    dataset = ZtoTauTauDataset(events_record, gpu_id)

    """
    for i in range(10):
        data = dataset.get(i)
        print(data)
    """

    if doplot:
        print("nx Graph")
        plt_nx_graph(dataset, outdir=outdir)
    

    train_dataset, val_dataset, test_dataset = dataset_split(dataset, config_dict.train_frac)

    train_loader, train_sampler, val_loader, test_loader = prepare_dataloaders(train_dataset, 
                                                                               val_dataset, 
                                                                               test_dataset, 
                                                                               config_dict.hparamdict.batchlen, 
                                                                               True)

    model_kwargs = get_model_kwargs()
    model = NuNet(**model_kwargs)

    print("Model parameters")
    total_params = count_model_params(model)
    
    print("Time to train / test / evaluate ...")
    nnTrainer = TrainerDDPtorchrun(gpu_id = gpu_id,
                                   hyper_param_dict = config_dict.hparamdict,
                                   model=model,
                                   trainloader=train_loader,
                                   sampler_train=train_sampler,
                                   valloader=val_loader,
                                   testloader=test_loader,
                                   num_out_classes=6,
                                   model_ckpt_path=ckptpath,
                                   dovalidate=doval)
    if dotrain:
        nnTrainer.train()

    saved_models = os.listdir(ckptpath)
    #if (len(saved_models) > 0):
    #    final_model_path = saved_models[-1]
    if dotest:
        #trues, preds = nnTrainer.test(os.path.join(ckptpath, final_model_path))
        print(f"model to test: {final_model_path}")
        trues, preds = nnTrainer.test(final_model_path)
        if doplot:
            pass


    # clean up
    destroy_process_group()



# Execute main function
if __name__ == "__main__":
    print("Checking GPU ...")
    print(f"is CUDA available? {torch.cuda.is_available()}")
    ndevices = torch.cuda.device_count()
    print(f"nGPUs: {ndevices}")
    gpu_id = torch.cuda.current_device()
    print(f"current device id: {gpu_id}")
    for id in [gpu_id]:
        print(f"current device name: {torch.cuda.get_device_name()}")


    parser = argparse.ArgumentParser(description='Train and evaluate GNN')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default="config.yaml",
                        help="contains the paths and initial h-parameters")
    parser.add_argument('-p',
                        '--plot',
                        action='store_true',
                        required=False,
                        default=False,
                        help="Plot features")

    args = parser.parse_args()


    config_ = args.config
    plot_ = args.plot

    main(config_, plot_)
