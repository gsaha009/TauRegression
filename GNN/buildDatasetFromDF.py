import os
import copy
from typing import Optional, Callable
import awkward as ak
import numpy as np
import pandas as pd
import itertools
import vector
import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

#from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import candidate

import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)

from torch_geometric.data import Data, Dataset, InMemoryDataset



class ZtoTauTauDataset(InMemoryDataset):
    def __init__(self,
                 root: str,
                 df_name: str, 
                 gpu_id: str, 
                 transform : Optional[Callable] = None, 
                 pre_transform : Optional[Callable] = None,
                 pre_filter : Optional[Callable] = None,
                 force_reload: bool = False,
    ):
        #self.load(self.processed_paths[0])
        print(f"constructor called")

        self.root = root
        print(self.root)
        self.df_name = df_name
        print(self.df_name)
        self.gpu_id = gpu_id
        self.nodetypes = ["tau", "jet"]
        self.nodefeats = ["pt", "eta", "phi", "mass",
                          "rawIsodR03", "rawMVAnewDM2017v2", "leadTkPtOverTauPt", "btagPNetB",
                          "rawPNetVSe", "rawPNetVSjet", "rawPNetVSmu", 
                          "charge_1", "charge_-1", 
                          "decayModePNet_0", "decayModePNet_1", "decayModePNet_2", 
                          "decayModePNet_10", "decayModePNet_11",
                          "npf", "ptfrac", "pf_dphi_pt_frac", "pf_deta_pt_frac"]
        self.globalfeats = ["nJet", "nPV", "HT", "MT_total", "pt_MET", "phi_MET", 
                            "covXX_MET", "covXY_MET", "covYY_MET", "sumEt_MET", 
                            "significance_MET"]
        self.targettypes = ["gentaunu"]
        self.targetfeats = ["px", "py", "pz"]

        print(f"mother constructor called")
        super(ZtoTauTauDataset, self).__init__(root,
                                               transform,
                                               pre_transform,
                                               pre_filter,
                                               force_reload)

        self.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return self.df_name

    def download(self):
        pass

    @property
    def processed_file_names(self):
        return 'data.pt'

    def _add_dummies(self, col_names):
        _df = self.df.copy(deep=True)
        print(f"df after deepcopy: \n{_df.head(10)}")
        for col_name in col_names:
            if col_name not in self.df.keys():
                print(f"WARNING: {col_name} not in df, adding zeros")
                _df[col_name] = 0.0

        return _df

    def _col_key_list(self, feat: str, max_particles: int, prefix: str) -> list :
        if max_particles > 0:
            return [f'{feat}_{prefix}_{i+1}' for i in range(max_particles)]
        else:
            return [f'{feat}_{prefix}']


    def _transform_from_dataframe(self):
        #print("_transform_from_dataframe called")
        tau_feat_names = []
        jet_feat_names = []
        for feat in self.nodefeats:
            print(f"Feature: {feat}")
            tau_feat_names += self._col_key_list(feat, 2, "tau")
            jet_feat_names += self._col_key_list(feat, 10, "jet")
        print(f"tau_feat_names: {tau_feat_names}")
        print(f"jet_feat_names: {jet_feat_names}")

        # new df with dummy cols
        #_df = self.df.copy(deep=True)
        #df = self._add_dummies(_df, tau_feat_names + jet_feat_names)
        df = self._add_dummies(tau_feat_names + jet_feat_names)
        #df = pd.concat([self.df.copy(deep=True), df_temp], axis=1) 
        print("df with all columns :--> ")
        print(list(df.keys()))
        print(df.head(10))


        node_dict = {}
        jet_mask = None
        njets_per_evt = None
        for i,feat in enumerate(self.nodefeats) :
            print(f"Node Feature: {feat}")
            tau_cols = self._col_key_list(feat, 2, "tau")
            #print(f"tau columns: {tau_cols}")
            jet_cols = self._col_key_list(feat, 10, "jet")
            #print(f"jet columns: {jet_cols}")

            tau_feats = ak.Array(df[tau_cols].values)
            #print(f"tau_feats: {tau_feats}")
            jet_feats = ak.Array(df[jet_cols].values)
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
            print(f"Global Feature: {feat}")
            global_feat = ak.Array(df[feat].values)
            global_dict[feat] = global_feat

            
        target_dict = {}
        for feat in self.targetfeats:
            print(f"Target Feature: {feat}")
            gentaunu_cols = self._col_key_list(feat, 2, "gentaunu")
            target_feat = ak.Array(df[gentaunu_cols].values)

            target_dict[feat] = target_feat

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
        #for node_key in ["pt", "eta", "phi", "mass", "charge"]:
            node_val = self.node_record[node_key]
            temp_x = ak.firsts(node_val[idx:idx+1], axis=0).to_numpy()
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
            target_val = self.target_record[target_key]
            temp_y = ak.firsts(target_val[idx:idx+1], axis=0).to_numpy()
            ys.append(temp_y[:2])
            aux_ys.append(temp_y[2:])
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
        print("process called")
        self.df = pd.read_hdf(self.raw_paths[0])
        ##self.df = pd.read_hdf(os.path.join(self.root, "raw/GluGluHToTauTauSMdata.h5"))
        zips = self._transform_from_dataframe()
        self.node_record = ak.zip(zips[0])
        self.global_record = ak.zip(zips[1])
        self.target_record = ak.zip(zips[2])
        datalist = []
        for idx in tqdm(range(len(self.df)), ascii=True):
            data = self.transform_to_torch(idx)
            #print(f"iter: {idx} ---> {data}")
            datalist.append(data)
            #sleep(0.01)
        self.save(datalist, 
                  os.path.join(self.processed_dir, 
                               "data.pt"))
        
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
