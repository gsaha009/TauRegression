import os
import awkward as ak
import numpy as np
import itertools

from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.nanoevents.methods import candidate

import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
from torch_geometric.data import Data, Dataset


class ZtoTauTauDataset(Dataset):
    def __init__(self, events_record: ak.Array, gpu_id: str):
        super(ZtoTauTauDataset, self).__init__()

        #self.root = os.path.join(os.getcwd(),"Datasets")
        self.node_record = events_record["node_feats"]
        self.target_record = events_record["target_feats"]
        self.global_record = events_record["global_feats"]
        globals = events_record["global_feats"].fields
        self.num_entries = ak.count(events_record["global_feats"][globals[0]])
        #self.num_entries = ak.count(ak.firsts(events_record.global_met_pt, axis=0))
        self.gpu_id = gpu_id
        
    def transform_to_torch(self, idx):
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
        #print(f"Device infos: {data.x.get_device()}, {data.y.get_device()}, {data.u.get_device()}, {data.edge_index.get_device()}, {data.edge_attr.get_device()}")

        return data

    
    def len(self):
        return self.num_entries

    def get(self, idx):
        return self.transform_to_torch(idx)
