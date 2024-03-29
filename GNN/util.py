import os
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable
import vector
import torch

class obj(object):
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [obj(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, obj(v) if isinstance(v, dict) else v)


class PlotUtil:
    def __init__(self, events_record: ak.Record, outdir: str):
        self.events_record = events_record
        self.outdir = outdir
        #self.record_fields = events_record.fields
        self.node_record = events_record["node_feats"]
        self.target_record = events_record["target_feats"]
        self.global_record = events_record["global_feats"]
        
    def plot_nodes(self, size: tuple):
        node_feats = self.node_record.fields
        nrows = len(node_feats)
        ncols = ak.max(ak.num(self.node_record[node_feats[0]], axis=1))
        
        fig, ax = plt.subplots(nrows,ncols,figsize=(size[0],size[1]))
        #plt.subplots_adjust(left    = 0.1,
        #                    right   = 0.6,
        #                    top     = 0.9,
        #                    bottom  = 0.1,
        #                    hspace  = 0.5,
        #                    wspace  = 0.4)
        for irow, feat in enumerate(node_feats):
            feat_arr = self.node_record[feat]
            #print(f"{feat}: {feat_arr}")
            for icol in range(ncols):
                temp = ak.ravel(feat_arr[:,icol:icol+1]).to_numpy()
                #print(temp)
                ax[irow,icol].hist(temp, bins=100, log=True, histtype="stepfilled", alpha=0.7, label=f'{feat}_{icol+1}')
                ax[irow,icol].legend()

        plt.tight_layout()
        #return plt
        plt.savefig(f"{self.outdir}/node_feats.png", dpi=350)

    def plot_targets(self, size: tuple):
        target_feats = self.target_record.fields
        nrows = len(target_feats)
        ncols = ak.max(ak.num(self.target_record[target_feats[0]], axis=1))
        
        fig, ax = plt.subplots(nrows,ncols,figsize=(size[0],size[1]))
        #plt.subplots_adjust(left    = 0.1,
        #                    right   = 0.6,
        #                    top     = 0.9,
        #                    bottom  = 0.1,
        #                    hspace  = 0.5,
        #                    wspace  = 0.4)
        for irow, feat in enumerate(target_feats):
            feat_arr = self.target_record[feat]
            #print(f"{feat}: {feat_arr}")
            for icol in range(ncols):
                temp = ak.ravel(feat_arr[:,icol:icol+1]).to_numpy()
                #print(temp)
                ax[irow,icol].hist(temp, bins=100, log=True, histtype="stepfilled", alpha=0.7, label=f'{feat}_{icol+1}')
                ax[irow,icol].legend()

        plt.tight_layout()
        plt.savefig(f"{self.outdir}/target_feats.png", dpi=350)
        #return plt


    def plot_globals(self):
        global_feats = self.global_record.fields
        nrows = int(np.ceil(np.sqrt(len(global_feats))))
        ncols = nrows
        plt.figure(figsize=(nrows*2,ncols*2))
        #fig, ax = plt.subplot(nrows,ncols,figsize=(12.5,9.5))
        
        for idx, feat in enumerate(global_feats):
            ax = plt.subplot(nrows,ncols,idx+1)
            feat_arr = self.global_record[feat]
            #print(f"{feat}: {feat_arr}")
            temp = ak.ravel(feat_arr).to_numpy()
            #print(temp)
            ax.hist(temp, bins=100, log=True, histtype="stepfilled", alpha=0.7, label=f'{feat}')
            ax.legend()
        plt.tight_layout()
        plt.savefig(f"{self.outdir}/global_feats.png", dpi=350)
        #return plt

def count_model_params(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    total_nontr_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            total_nontr_params += parameter.numel()
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    print(f"Total Non-Trainable Params: {total_nontr_params}")
    return total_params

def getE(px: np.array, py:np.array, pz:np.array)->np.array:
    return np.sqrt(px**2 + py**2 + pz**2)

def getp4(out: torch.Tensor, idx: int):
    idx = idx - 1 if idx == 1 else 3
    out = out.cpu()
    px = out[:, idx:idx+1].numpy().reshape(-1)
    py = out[:, idx+1:idx+2].numpy().reshape(-1)
    pz = out[:, idx+2:idx+3].numpy().reshape(-1)
    E = getE(px, py, pz)

    return vector.array(
        {
            "px": px,
            "py": py,
            "pz": pz,
            "E": E,
        }
    )

def plteval(y_true: torch.Tensor, y_pred: torch.Tensor, path: str) -> None:
    plt.figure(figsize=(20,12))
    ntargets = y_true.shape[-1]
    for i in range(ntargets):
        ax = plt.subplot(int(ntargets/3),3,i+1)
        
        temp_true = y_true[:,i:i+1].cpu().numpy().reshape(-1)
        temp_pred = y_pred[:,i:i+1].cpu().numpy().reshape(-1)
        
        ax.hist(temp_true, 100, range=[-50.0,50.0], histtype="stepfilled", alpha=0.7, label='True')
        ax.hist(temp_pred, 100, range=[-50.0,50.0], histtype="stepfilled", alpha=0.7, label='Pred')
    #ax.set_title(f"{key}")
    #ax.set_xlabel(f'''{key.split('_')[-1]}''')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'output_regressed.png'), dpi=300)
        
def plteval_v2(y_true: torch.Tensor, y_pred: torch.Tensor, path: str) -> None:
    y_true = y_true.cpu()
    y_pred = y_pred.cpu()

    true_nu1_p4 = getp4(y_true, 1)
    true_nu2_p4 = getp4(y_true, 2)
    regr_nu1_p4 = getp4(y_pred, 1)
    regr_nu2_p4 = getp4(y_pred, 2)
    
    nu1_dict = {
        'nu1_pt_true': true_nu1_p4.pt,
        'nu1_pt_regr': regr_nu1_p4.pt,
        'nu1_eta_true': true_nu1_p4.eta,
        'nu1_eta_regr': regr_nu1_p4.eta,
        'nu1_phi_true': true_nu1_p4.phi,
        'nu1_phi_regr': regr_nu1_p4.phi,
        'nu1_p_true': true_nu1_p4.p,
        'nu1_p_regr': regr_nu1_p4.p,
    }
    nu2_dict = {
        'nu2_pt_true': true_nu2_p4.pt,
        'nu2_pt_regr': regr_nu2_p4.pt,
        'nu2_eta_true': true_nu2_p4.eta,
        'nu2_eta_regr': regr_nu2_p4.eta,
        'nu2_phi_true': true_nu2_p4.phi,
        'nu2_phi_regr': regr_nu2_p4.phi,
        'nu2_p_true': true_nu2_p4.p,
        'nu2_p_regr': regr_nu2_p4.p,
    }

    plt.figure(figsize=(16,10))
    items = ['pt', 'eta', 'phi', 'p']
    for idx, key in enumerate(items):
        idx=idx+1
        ax = plt.subplot(2,2,idx)
        arr_true = nu1_dict[f'nu1_{key}_true'].reshape(-1)
        arr_pred = nu1_dict[f'nu1_{key}_regr'].reshape(-1)
        
        #print(arr)
        minmax = [-3.2,3.2] if np.min(arr_true) < 0 else [0,60]
        ax.hist(arr_true, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
        ax.hist(arr_pred, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed')
        ax.set_title(f"{key}")
        ax.set_xlabel(f"""{key.split('_')[-1]}""")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'output_nu_1.png'), dpi=300)
    plt.clf()
    plt.figure(figsize=(16,10))
    for idx, key in enumerate(items):
        idx=idx+1
        ax = plt.subplot(2,2,idx)
        arr_true = nu2_dict[f'nu2_{key}_true'].reshape(-1)
        arr_pred = nu2_dict[f'nu2_{key}_regr'].reshape(-1)
        
        #print(arr)
        minmax = [-3.2,3.2] if np.min(arr_true) < 0 else [0,60]
        ax.hist(arr_true, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='True')
        ax.hist(arr_pred, 70, range=minmax, histtype="stepfilled", alpha=0.7, label='Regressed')
        ax.set_title(f"{key}")
        ax.set_xlabel(f"""{key.split('_')[-1]}""")
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path,'output_nu_2.png'), dpi=300)
