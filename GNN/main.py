# python general import
import os
import glob
import yaml
import datetime
import argparse

# ext array library
import awkward as ak
import numpy as np
import pandas as pd
from typing import Optional

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns

# torch and PyG
import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, NNConv, global_add_pool, global_mean_pool, MetaLayer, MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import to_networkx

# graph viewer
import networkx as nx
#from torch_scatter import scatter

# user import
from util import obj, PlotUtil, plothistory, count_model_params, plteval, plteval_v2
#from buildDataset import ZtoTauTauDataset
from buildDatasetFromDF import ZtoTauTauDataset
from buildModel import NuNet#, get_model_kwargs
from train import Trainer

from mainutil import *
try:
    import cPickle as pickle
except ImportError:  # Python 3.x
    import pickle
    


# Main function
def main(configfile: str, doplot: bool, gpu_id: int):

    datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    config_dict = obj(load_config(configfile))

    torch.manual_seed(config_dict.torch_seed)
    
    #input_path = config_dict.inputs.datapath\
    input_path = config_dict.rawdata_path
    #node_file = f"{input_path}/{config_dict.inputs.node_file}"
    #target_file = f"{input_path}/{config_dict.inputs.target_file}"
    #global_file = f"{input_path}/{config_dict.inputs.global_file}"
    _df_name = config_dict.df_name
    raw_data = os.path.join(input_path, _df_name)
    _proc_data_name = config_dict.procdata_name
    #_df = f"{input_path}/{config_dict.inputs.dataframe}"
    #df = pd.read_hdf(_df)

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
    
    #node_info = ak.from_parquet(node_file)
    #target_info = ak.from_parquet(target_file)
    #global_info = ak.from_parquet(global_file)

    #events_record = {}
    #events_record["node_feats"] = node_info
    #events_record["target_feats"] = target_info
    #events_record["global_feats"] = global_info

    
    #if doplot:
    #    print("Plotting")
    #    plotter = PlotUtil(events_record, outdir)
    #    plotter.plot_nodes((40,40))
    #    plotter.plot_targets((20,12))
    #    plotter.plot_globals()

    node_types = config_dict.nodetypes
    node_feats = config_dict.nodefeats
    target_types = config_dict.targettypes
    target_feats = config_dict.targetfeats
    global_feats = config_dict.globalfeats

    print("Preparing PyG Dataset")
    #dataset = ZtoTauTauDataset(df, gpu_id)
    #dataset = ZtoTauTauDataset(gpu_id, _df_name, root="data/")
    dataset = ZtoTauTauDataset("/pbs/home/g/gsaha/Work/TauRegression/GNN/data/", 
                               #_df_name, 
                               raw_data,
                               _proc_data_name,
                               gpu_id,
                               outdir,
                               node_types,
                               node_feats,
                               #target_types,
                               target_feats,
                               global_feats,
                               do_plot=doplot,
                               force_reload=True)

    for i in range(10):
        data = dataset.get(i)
        print(data)

    if doplot:
        print("nx Graph")
        plot_nx_graph(dataset, outdir)
    
    train_dataset, val_dataset, test_dataset = dataset_split(dataset, config_dict.train_frac)
    train_loader, train_sampler, val_loader, test_loader = prepare_dataloaders(train_dataset, 
                                                                               val_dataset, 
                                                                               test_dataset, 
                                                                               config_dict.hparamdict.batchlen, 
                                                                               False)
    model_kwargs = {
        "node_feat_size": len(node_feats),
        "global_feat_size": len(global_feats),
        "num_classes": len(target_feats),
        "depth": 2,
        "dropout": True
    }
    #model_kwargs = get_model_kwargs()
    print(model_kwargs)
    model = NuNet(**model_kwargs)

    print("Model parameters")
    total_params = count_model_params(model)
    
    print("Time to train / test / evaluate ...")
    nnTrainer = Trainer(gpu_id = gpu_id,
                        hyper_param_dict = config_dict.hparamdict,
                        model=model.to(gpu_id),
                        trainloader=train_loader,
                        valloader=val_loader,
                        testloader=test_loader,
                        num_out_classes=model_kwargs["num_classes"],
                        model_ckpt_path=ckptpath,
                        dovalidate=doval)

    if dotrain:
        history = nnTrainer.train()
        #print(history.batch.loss)
        #print(history.epoch.loss.train)
        #print(history.epoch.loss.val)
        #print(history.epoch.accuracy.train)
        #print(history.epoch.accuracy.val)
        #print(history.epoch.LR)
        # save history as pickle
        with open(os.path.join(outdir,'data.p'), 'wb') as fp:
            pickle.dump(history, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if doplot:
            print("Plotting model performance with train and validation dataset")
            plothistory(obj(history), outdir)


        
    saved_models = os.listdir(ckptpath)
    #if (len(saved_models) > 0):
    #    final_model_path = saved_models[-1]
    if dotest:
        #trues, preds = nnTrainer.test(os.path.join(ckptpath, final_model_path))
        trues, preds = nnTrainer.test(final_model_path)
        print(trues, preds)
        plteval(trues, preds, outdir)
        plteval_v2(trues, preds, outdir)

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

    #gpu_id = torch.device("cuda", gpu_id)

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

    config_  = args.config
    plot_    = args.plot

    main(config_, plot_, gpu_id)
