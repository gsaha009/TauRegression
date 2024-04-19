# python general import
import os
import glob
import yaml
import datetime
import argparse
import logging

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
    

def setup_logger(log_file):
    # Create a logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# Main function
def main(configfile: str):
    datetime_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logger = setup_logger(os.path.join("logs", f"Runlog_{datetime_tag}.log"))

    config_dict = obj(load_config(configfile))
    torch.manual_seed(config_dict.torch_seed)
    logger.info(f"torch manual seed: {config_dict.torch_seed}")
    input_path      = config_dict.rawdata_path
    _df_name        = config_dict.df_name
    raw_data        = os.path.join(input_path, _df_name)
    logger.info(f"raw dataframe : {raw_data}")
    _proc_data_name = config_dict.procdata_name
    logger.info(f"processed PyG dataset name : {_proc_data_name}")

    doplot   = config_dict.plot.cmd
    logger.info(f"Plot? : {doplot}")
    donorm   = config_dict.norm.cmd
    logger.info(f"Norm? : {donorm}")
    dotrain  = config_dict.train.cmd
    logger.info(f"Train? : {dotrain}")
    doval    = config_dict.val.cmd
    logger.info(f"Validation? : {doval}")
    dotest   = config_dict.test.cmd
    logger.info(f"Test? : {dotest}")
    final_model_path = config_dict.final_model_path
    logger.info(f"Final model path : {final_model_path}")
    doeval   = config_dict.evaluate.cmd
    logger.info(f"Evaluation? : {doeval}")

    outputpath = config_dict.outputpath
    modelpath  = config_dict.modelpath
    ckptpath   = f"{modelpath}_{datetime_tag}"
    outdir     = f"{outputpath}_{datetime_tag}"
    if dotrain:
        logger.info(f"Creating dir to save models : {ckptpath}")
        os.makedirs(ckptpath, exist_ok=True)
        logger.info(f"Creating dir to save ouputs : {outdir}")
        os.makedirs(outdir, exist_ok=True)
    else:
        if dotest or doeval:
            outdir = os.path.join(os.path.dirname(outputpath), final_model_path.split('/')[-2].replace("TrainedModels","Output"))
            logger.info(f"Using the old outdir following trained model dir: {outdir}")
            if os.path.exists(outdir):
                logger.info(f"{outdir} exists ...")
            else:
                logger.warning(f"{outdir} not found")
                raise RuntimeError("check model path")


    logger.info("Checking GPU ...")
    logger.info(f"is CUDA available? {torch.cuda.is_available()}")
    ndevices = torch.cuda.device_count()
    logger.info(f"nGPUs: {ndevices}")
    gpu_id = torch.cuda.current_device()
    logger.info(f"current device id: {gpu_id}")
    for id in [gpu_id]:
        logger.info(f"current device name: {torch.cuda.get_device_name()}")

    #gpu_id = torch.device("cuda", gpu_id)

    """
    if dotest or doeval: 
         outdir = final_model_path.split('/')[-2].replace("TrainedModels","Output")
         if os.path.exists(outdir):
             print(f"{outdir} exists ...")
         else:
             raise RuntimeError("check model path")
    else:
        ckptpath = f"{config_dict.modelpath}_{datetime_tag}"
        os.makedirs(ckptpath, exist_ok=True)
    """

    node_types = config_dict.nodetypes
    node_feats = config_dict.nodefeats
    target_types = config_dict.targettypes
    target_feats = config_dict.targetfeats
    global_feats = config_dict.globalfeats

    logger.info("Preparing PyG Dataset ===>")
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
                               do_norm=donorm,
                               do_plot=doplot,
                               force_reload=True)

    for i in range(10):
        data = dataset.get(i)
        logger.info(data)

    if doplot:
        logger.info("Plotting nx Graph ---> ")
        plot_nx_graph(dataset, outdir)
    
    train_dataset, val_dataset, test_dataset = dataset_split(dataset, config_dict.train_frac)
    train_loader, train_sampler, val_loader, test_loader = prepare_dataloaders(train_dataset, 
                                                                               val_dataset, 
                                                                               test_dataset, 
                                                                               config_dict.hparamdict.batchlen, 
                                                                               False)
    model_kwargs = {
        "node_feat_size": len(node_feats.numerical) + len(node_feats.categorical),
        "global_feat_size": len(global_feats),
        "num_classes": len(target_feats),
        "depth": 2,
        "dropout": True
    }
    #model_kwargs = get_model_kwargs()
    logger.info(model_kwargs)
    model = NuNet(**model_kwargs)

    logger.info("Model parameters")
    total_params = count_model_params(model)
    
    logger.info("Time to train / test / evaluate ...")
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
            logger.info("Plotting model performance with train and validation dataset")
            plothistory(obj(history), outdir)

        saved_models = os.listdir(ckptpath)
        if dotest or doeval:
            if (len(saved_models) > 0):
                final_model_path = os.path.join(ckptpath, saved_models[-1])
            else:
                raise RuntimeWarning("no saved models")
    if dotest:
        #trues, preds = nnTrainer.test(os.path.join(ckptpath, final_model_path))
        logger.info(f"test using model: {final_model_path}")
        trues, preds = nnTrainer.test(final_model_path)
        logger.info(f"true targets : {trues}") 
        logger.info(f"pred targets : {preds}")
        plteval(trues, preds, outdir)
        plteval_v2(trues, preds, outdir)

    if doeval:
        logger.info("Time to evaluate ...")
        eval_loader, _ = prepare_dataloader(dataset,
                                            config_dict.hparamdict.batchlen,
                                            False,
                                            False)
        #nnEvaluator = Trainer(gpu_id = gpu_id,
        #                      hyper_param_dict = config_dict.hparamdict,
        #                      model=model.to(gpu_id),
        #                      trainloader=train_loader,
        #                      valloader=val_loader,
        #                      testloader=eval_loader,
        #                      num_out_classes=model_kwargs["num_classes"],
        #                      model_ckpt_path=ckptpath,
        #                      dovalidate=doval)
        trues, preds = nnTrainer.evaluate(final_model_path, eval_loader)
        logger.info(f"true targets : {trues}")
        logger.info(f"pred targets : {preds}")
        np_y_pred = plteval(trues, preds, outdir)
        plteval_v2(trues, preds, outdir)
        logger.info("Getting the modified dataframe ... ")
        df_processed_file = os.path.join(outdir, f"ypreds_{datetime_tag}.h5")
        df_out = pd.DataFrame(np_y_pred, columns=target_feats)
        logger.info(df_out.head(10))
        logger.info(f"saving the predicted targets in a df: {df_processed_file}")
        df_out.to_hdf(df_processed_file, key='df', mode='w')
        logger.info("Done")
        #print("Saving the output to a dataframe ... ")
        

# Execute main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate GNN')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default="config.yaml",
                        help="contains the paths and initial h-parameters")

    args = parser.parse_args()

    config_  = args.config

    #main(config_, gpu_id)
    main(config_)
