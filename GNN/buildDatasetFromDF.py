import os
import copy
from typing import Optional, Callable
import awkward as ak
import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler
import logging
logger = logging.getLogger('main')
import vector
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import candidate

import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
logger.info(torch.__version__)

from torch_geometric.data import Data, Dataset, InMemoryDataset
from util import PlotUtil, plotdf
from mainutil import normalise_column



class ZtoTauTauDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 df_name: str,
                 proc_name: str,
                 gpu_id: str,
                 plotdir: str,
                 node_types: list,
                 node_feats: list,
                 #target_types: list,
                 target_feats: list,
                 global_feats: list,
                 do_norm: bool = True,
                 do_plot: bool = True,
                 transform : Optional[Callable] = None, 
                 pre_transform : Optional[Callable] = None,
                 pre_filter : Optional[Callable] = None,
                 force_reload: bool = False,
    ):
        #self.load(self.processed_paths[0])
        #print(f"constructor called")
        
        self.root = root
        print(self.root)
        self.df_name = df_name
        print(self.df_name)
        self.proc_name = f"Norm_{proc_name}" if do_norm else f"Raw_{proc_name}" 
        self.gpu_id = gpu_id
        self.nodetypes = node_types
        self.nodefeats = node_feats
        #self.targettypes = target_types
        self.targetfeats = target_feats
        self.globalfeats = global_feats
        
        self.plotdir = plotdir
        self.donorm = do_norm
        self.doplot = do_plot

        #print(f"mother constructor called")
        super(ZtoTauTauDataset, self).__init__(root,
                                               transform,
                                               pre_transform,
                                               pre_filter,
                                               force_reload)

        self.load(self.processed_paths[0])

    #@property
    #def raw_dir(self) -> str:
    #    return os.path.join(self.root, 'raw', 'norm')
    
    @property
    def raw_file_names(self):
        return self.df_name

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return self.proc_name

    def _add_dummies(self, col_names):
        _df = self.df.copy(deep=True)
        logger.info(f"df after deepcopy: \n{_df.head(10)}")
        for col_name in col_names:
            if col_name not in self.df.keys():
                logger.warning(f"{col_name} not in df, adding zeros")
                _df[col_name] = 0.0

        _df = _df.fillna(0)
        return _df

    def _col_key_list(self, feat: str, max_particles: int, prefix: str) -> list :
        if max_particles > 0:
            return [f'{feat}_{prefix}_{i+1}' for i in range(max_particles)]
        else:
            if prefix == None:
                return [f'{feat}']
            else:
                return [f'{feat}_{prefix}']


    def _transform_from_dataframe(self):
        #print("_transform_from_dataframe called")
        tau_cat_feat_names = []
        jet_cat_feat_names = []
        tau_num_feat_names = []
        jet_num_feat_names = []

        allnodefeats = self.nodefeats.numerical + self.nodefeats.categorical

        for feat in self.nodefeats.numerical:
            logger.info(f"Feature: {feat}")
            tau_num_feat_names += self._col_key_list(feat, 2, "tau")
            jet_num_feat_names += self._col_key_list(feat, 10, "jet")
            
        for feat in self.nodefeats.categorical:
            logger.info(f"Feature: {feat}")
            tau_cat_feat_names += self._col_key_list(feat, 2, "tau")
            jet_cat_feat_names += self._col_key_list(feat, 10, "jet")
            
            
        tau_feat_names = tau_num_feat_names + tau_cat_feat_names
        jet_feat_names = jet_num_feat_names + jet_cat_feat_names

        #for feat in allnodefeats:
        #    print(f"Feature: {feat}")
        #    tau_feat_names += self._col_key_list(feat, 2, "tau")
        #    jet_feat_names += self._col_key_list(feat, 10, "jet")
        logger.info(f"tau_feat_names: {tau_feat_names}")
        logger.info(f"jet_feat_names: {jet_feat_names}")

        # new df with dummy cols
        #_df = self.df.copy(deep=True)
        #df = self._add_dummies(_df, tau_feat_names + jet_feat_names)
        df = self._add_dummies(tau_feat_names + jet_feat_names)
        #df = pd.concat([self.df.copy(deep=True), df_temp], axis=1)

        logger.info("df with all columns :--> ")
        logger.info(list(df.keys()))
        logger.info(f"\n{df.head(10)}")
        if self.doplot:
            #logger.info("Plotting the modified [raw] dataframe ... ")
            #plotdf(df, os.path.join(self.plotdir, f"{self.df_name}_Modified_Raw.pdf"))
            logger.info(f"Plotting the modified [raw] dataframe in {self.plotdir}... ")
            plotdf(df, os.path.join(self.plotdir, f"RawModified_{os.path.basename(self.df_name).split('.')[0]}.pdf"))

                   
        if self.donorm:
            logger.info("Normalising df: 1. Numerical feats only, 2. Not any target variables")
            colsAll = list(df.keys())
            colsToNorm = self.globalfeats + tau_num_feat_names + tau_cat_feat_names
            
            logger.info(f"Keys to normalise: {colsToNorm}")
            colsNotToNorm = np.setdiff1d(np.array(colsAll), np.array(colsToNorm)).tolist()
            dfToNorm    = df[colsToNorm]
            #from IPython import embed; embed()
            logger.info("Before norm :--> ")
            logger.info(dfToNorm.head(10))
            dfNotToNorm = df[colsNotToNorm]
            #scaler = MinMaxScaler()
            #dfToNorm = scaler.fit_transform(dfToNorm)
            dfToNorm = dfToNorm.apply(normalise_column)
            logger.info("After norm :--> ")
            logger.info(dfToNorm.head(10))
            df = pd.concat([dfToNorm, dfNotToNorm], axis=1)

            logger.info("df with all columns [after normalisation] :--> ")
            #print(list(df.keys()))
            logger.info(df.head(10))
        

        node_dict = {}
        jet_mask = None
        njets_per_evt = None
        #for i,feat in enumerate(self.nodefeats) :
        for i,feat in enumerate(allnodefeats) :
            logger.info(f"Node Feature: {feat}")
            tau_cols = self._col_key_list(feat, 2, "tau")
            #print(f"tau columns: {tau_cols}")
            jet_cols = self._col_key_list(feat, 10, "jet")
            #print(f"jet columns: {jet_cols}")
            tau_col_vals = df[tau_cols].values
            assert not np.any(np.isnan(tau_col_vals))
            #temp_tau_col_vals = tau_col_vals.flatten()
            #print(np.isnan(temp_tau_col_vals), np.sum(temp_tau_col_vals))
            tau_feats = ak.Array(tau_col_vals)

            jet_col_vals = df[jet_cols].values
            #print(f"jet: {list(jet_col_vals)}")
            #assert not np.any(np.isnan(jet_col_vals))
            jet_feats = ak.Array(jet_col_vals)
            #print(f"jet_feats: {jet_feats}")
            if i == 0:
                jet_mask = jet_feats > 0.0
                #njets_per_evt = ak.sum(mask, axis=1)
            #print(f"jet_mask: {jet_mask}")
            #jet_feats = jet_feats[jet_mask]
            jet_feats = ak.drop_none(ak.mask(jet_feats, jet_mask))
            #jet_feats = ak.unflatten(ak.flatten(jet_feats), njets_per_evt)
            #print(f"jet feats after masking: {jet_feats}")
            node_feats = ak.concatenate([tau_feats, jet_feats], axis=1)
            #print(f"node_feats: {node_feats}")
            node_dict[feat] = node_feats
            
            
        global_dict = {}
        for feat in self.globalfeats:
            logger.info(f"Global Feature: {feat}")
            global_feat = ak.Array(df[feat].values)
            global_dict[feat] = global_feat

            
        target_dict = {}
        for feat in self.targetfeats:
            logger.info(f"Target Feature: {feat}")
            #gentaunu_cols = self._col_key_list(feat, 2, "gentaunu")
            #gentaunu_cols = self._col_key_list(feat, 2, None)
            target_feat = ak.Array(df[feat].values)

            target_dict[feat] = target_feat

        #extra_target_col = "phicp"
        #temp_feat = ak.Array(df[extra_target_col].values)
        #target_dict[extra_target_col] = target_feat
        logger.info(f"Saving the modified dataframe in {self.plotdir} ...")
        logger.info(self.df_name)
        df.to_hdf(os.path.join(self.plotdir, f"Modified_{os.path.basename(self.df_name)}"), key='df', mode='w')
        logger.info(" ... Done")
        if self.doplot and self.donorm:
            logger.info(f"Plotting the modified [norm] dataframe in {self.plotdir}... ")
            plotdf(df, os.path.join(self.plotdir, f"NormModified_{os.path.basename(self.df_name).split('.')[0]}.pdf"))

        return (node_dict, global_dict, target_dict)

        
    def transform_to_torch(self, idx):
        #print("transform_to_torch called")
        xs = []
        ys = []
        aux_ys = []
        us = []

        nodes_p4 = ak.zip(
            {
                "pt": self.node_record["pt"][idx:idx+1],
                "eta": self.node_record["eta"][idx:idx+1],
                "phi": self.node_record["phi"][idx:idx+1],
                "mass": self.node_record["mass"][idx:idx+1],
            },
            with_name="PtEtaPhiMCandidate",
            behavior=candidate.behavior,
        )
        nodes_p4 = ak.firsts(nodes_p4, axis=0)
        #print(f"nodes_p4: {nodes_p4}")
        
        for node_key in self.node_record.fields:
            #print(f"node feat: {node_key}")
            node_val = self.node_record[node_key]
            temp_x = ak.firsts(node_val[idx:idx+1], axis=0).to_numpy()
            #print(f"--- val: {temp_x}")
            xs.append(temp_x)

        node_feats = np.stack(xs).T
        #print(f"node_feats: {node_feats}")

        node_x = torch.tensor(node_feats).to(torch.float32)
        node_x = node_x.to(self.gpu_id)
        #print(f"Tensor: node_feats: {node_x}")

        # preparing edges
        nodeidxs = list(range(node_feats.shape[0]))
        edgeidxs = []
        for pair in list(itertools.combinations(nodeidxs, 2)):
            pair = list(pair)
            edgeidxs.append(pair)
            edgeidxs.append(pair[::-1])
        edgeidxs = torch.tensor(edgeidxs).to(torch.long)
        edgeidxs = edgeidxs.to(self.gpu_id)
        #print(f"edge_idxs: {edgeidxs}")

        src, dst = edgeidxs[:,0].cpu().detach().numpy(), edgeidxs[:,1].cpu().detach().numpy()
        #print(f"src: {src}, dst: {dst}")
        
        edges_dr = torch.tensor(nodes_p4[src].delta_r(nodes_p4[dst])).to(torch.float32)
        edges_attr = edges_dr.reshape(-1,1)
        edges_attr = edges_attr.to(self.gpu_id)
        #print(f"edges_dr : {edges_attr}")
        
        for target_key in self.target_record.fields:
            #print(f"target feat: {target_key}")
            target_val = self.target_record[target_key]
            #temp_y = ak.firsts(target_val[idx:idx+1], axis=0).to_numpy()
            temp_y = target_val[idx:idx+1].to_numpy()
            #ys.append(temp_y[:2])
            #aux_ys.append(temp_y[2:])
            #print(f"--- val: {temp_y}")
            ys.append(temp_y)
            aux_ys.append(temp_y)

        target_feats = np.stack(ys).T
        #print(f"target_feats: {target_feats}")
        target_y = torch.tensor(target_feats).to(torch.float32)
        target_y = target_y.reshape(1,int(target_y.shape[0]*target_y.shape[1]))
        target_y = target_y.to(self.gpu_id)
        #print(f"Tensor: target_feats: {target_y}")

        aux = np.stack(aux_ys).T
        aux = torch.tensor(aux).to(torch.float32)
        aux = aux.to(self.gpu_id)
        #print(f"aux: {aux}")

        for global_key in self.global_record.fields:
            global_val = self.global_record[global_key]
            temp_x = global_val[idx:idx+1].to_numpy()
            us.append(temp_x)
        #print(us)
        #print(f"global_feats: {us}")
        global_u = torch.tensor(np.array(us)).to(torch.float32).reshape(1,-1)
        global_u = global_u.to(self.gpu_id)
        #print(f"Tensor: global_feats: {global_u}")


        #print(f"Types: {type(node_x)}\n{type(edgeidxs)}\n{type(edges_attr)}\n{type(target_y)}\n{type(global_u)}\n{type(aux)}")
        data = Data(x=node_x,
                    edge_index = edgeidxs.t().contiguous(),
                    edge_attr = edges_attr,
                    y=target_y,
                    u=global_u)
        data.aux_gentaus = aux

        #print(f"{type(data)}")
        #print(f"D A T A : {data.x}, {data.y}, {data.u}, {data.edge_index}, {data.edge_attr}")

        return data


    def process(self):
        from tqdm import tqdm
        from time import sleep
        logger.info("process called")
        logger.info(self.raw_paths)
        logger.info(f"self.raw_paths[0]: {self.raw_paths[0]}")
        
        #self.df = pd.read_hdf(self.raw_paths[0])
        self.df = pd.read_hdf(self.raw_file_names)
        logger.info(f"raw data: {self.df}")
        #self.df = pd.read_hdf("/pbs/home/g/gsaha/work/TauRegression/GNN/data/raw/GluGluHToTauTauSMdata_00_11.h5")
        zips = self._transform_from_dataframe()
        #self.df_modified = zips[0]

        self.node_record = ak.zip(zips[0])
        self.global_record = ak.zip(zips[1])
        self.target_record = ak.zip(zips[2])
        if self.doplot:
            logger.info(" ===> Plotting : Start ===> ")
            plotter = PlotUtil(self.node_record, 
                               self.target_record,
                               self.global_record,
                               self.plotdir)
            plotter.plot_nodes((40,40))
            plotter.plot_targets((20,12))
            plotter.plot_globals()
            logger.info(" ===> Plotting : End ===> ")
        datalist = []
        for idx in tqdm(range(len(self.df)), ascii=True):
            data = self.transform_to_torch(idx)
            #print(f"iter: {idx} ---> {data}")
            datalist.append(data)
            #sleep(0.01)
        logger.info(f"Saving the processed PytorchGeometric Dataset in {self.processed_dir} ...")
        self.save(datalist,
                  os.path.join(self.processed_dir,
                               self.proc_name))
        logger.info(" ... Done")
            
    """
    def len(self):
        df = pd.read_hdf(self.raw_paths[0])
        #return len(self.processed_file_names)
        return len(df)
        

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data
        #    #return self.transform_to_torch(idx)
    """
