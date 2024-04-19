import os
import fnmatch
import awkward as ak
import numpy as np

from tqdm.notebook import tqdm
from typing import Callable, Union
import itertools

import torch
import torch.nn as nn
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)
from torch.nn import LayerNorm, Dropout
from torch_geometric.nn import GCNConv, GATConv, global_add_pool, global_mean_pool, MessagePassing
import torch.nn.functional as F


# ------------------------------------------------------------------------------------ #
#                                    Node model                                        #
# Use multi-head Graph Attention which includes the DeltaR as edge attr. Finally, the  #
# raw node features are embedded into 2x dimensional embedding                         #
# ------------------------------------------------------------------------------------ #
class GATConvBlock(torch.nn.Module):
    def __init__(self, dim_in: int, dim_h: int, dim_out: int, heads: int = 5):
        super(GATConvBlock, self).__init__()
        
        self.gat1 = GATConv(dim_in, dim_h, heads=heads)
        #self.norm1 = LayerNorm(dim_h*heads)
        self.norm1 = nn.BatchNorm1d(dim_h*heads)
        self.drop = nn.Dropout(p=0.2)
        self.gat2 = GATConv(dim_h*heads, dim_out, heads=1)
        #self.norm2 = LayerNorm([dim_out])
        self.norm2 = nn.BatchNorm1d(dim_out)
        #self.gat2 = GATConv(dim_in, dim_out, heads=1)
        
    def forward(self, x, edge_index, edge_attr):
        x = self.gat1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.selu(x)
        x = self.drop(x)
        x = self.gat2(x, edge_index)
        x = self.norm2(x)
        x = F.selu(x)
        x = self.drop(x)
        
        return x


# ------------------------------------------------------------------------------------ #
#                                    Edge model                                        #
# Similar to ParticleNet Edge-Convolution. The modification is the concatenation of xi #
# and xj instead of (xi -xj), xi. The steps are the following:                         #
# - build_mlp: a stack of lin layers with non-linear activation used at the edge conv  #
# - EdgeConv_lrp: A Message Passing layer to demonstrate the behavior of operation     #
# - EdgeConvBlock: MLP in edgeConv layer and then the conv layer to form nn module     #
# ------------------------------------------------------------------------------------ #

# build a mlp for edge convolution
def build_mlp(in_size, layer_size, depth):
    layers = []

    layers.append(nn.Linear(in_size * 2, layer_size))
    layers.append(nn.BatchNorm1d(layer_size))
    layers.append(nn.SELU())
    layers.append(nn.Dropout(p=0.2)) # new

    for i in range(depth):
        layers.append(nn.Linear(layer_size, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.SELU())
        layers.append(nn.Dropout(p=0.2)) # new

    return nn.Sequential(*layers)

# build the conv layer
class EdgeConv_lrp(MessagePassing):
    def __init__(self, nn: Callable, aggr: str = "max", **kwargs):
        super(EdgeConv_lrp, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        
    def forward(self, x, edge_index):
        return (
            self.propagate(edge_index, x=x),
            self.edge_activations
        )
    
    def message(self, x_i, x_j):
        self.edge_activations = self.nn(torch.cat([x_i, x_j], dim=-1))
        return self.nn(torch.cat([x_i, x_j], dim=-1))

    def __repr__(self):
        return f"{self.__class__.__name__}(nn={self.nn})"
    
# build edge model
class EdgeConvBlock(torch.nn.Module):
    def __init__(self, in_size, layer_size, depth):
        super(EdgeConvBlock, self).__init__()
        
        edge_mlp = build_mlp(in_size=in_size, layer_size=layer_size, depth=depth)
        self.edge_conv = EdgeConv_lrp(edge_mlp, aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


# ------------------------------------------------------------------------------------ #
#                                     Full Graph Model                                 #
# build the graph model                                                                #
# GAT using node and original edge feat to embed more info                             #
# Edge convolution using those node embeddings                                         #
# ------------------------------------------------------------------------------------ #

class NuNet(torch.nn.Module):
    def __init__(self, 
                 node_feat_size,
                 global_feat_size,
                 num_classes,
                 depth=2,
                 dropout=False):
        super(NuNet, self).__init__()
        
        self.node_feat_size = node_feat_size
        self.global_feat_size = global_feat_size
        self.num_classes = num_classes

        self.num_edge_conv_blocks = 2
        
        self.kernel_sizes = [2*self.node_feat_size, 256, 256]
        #self.kernel_sizes = [2*self.node_feat_size, 32, 64]
        self.input_sizes = np.cumsum(self.kernel_sizes)
        self.fc_size = 256

        if dropout:
            self.dropout = 0.2
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout = None
            
        # node conv block
        self.nodeblock = GATConvBlock(self.node_feat_size, 2*self.node_feat_size, 2*self.node_feat_size)
        
        # define the edgeconvblocks
        self.edge_conv_blocks = torch.nn.ModuleList()
        for i in range(self.num_edge_conv_blocks):
            self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i+1], depth=depth))

        # define the fully connected networks (post-edgeconvs)
        self.fc1 = nn.Linear(self.input_sizes[-1]+global_feat_size, self.fc_size)
        self.norm1 = nn.BatchNorm1d(self.input_sizes[-1]+global_feat_size)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)
        self.norm2 = nn.BatchNorm1d(self.fc_size)
        
    def forward(self, batch):
        x  = batch.x
        y  = batch.y
        u  = batch.u
        ei = batch.edge_index
        ea = batch.edge_attr
        batch = batch.batch
        
        x = self.nodeblock(x, ei, ea)
        for i in range(self.num_edge_conv_blocks):
            out, _ = self.edge_conv_blocks[i](x, ei)
            x = torch.cat((out, x), dim=1)
            
        x = global_mean_pool(x, batch)
        #print(x.shape)
        #print(u.shape)
        x = torch.cat((x, u), dim=1)
        x = self.norm1(x) # new        
        x = self.fc1(x)
        x = F.selu(x)
        #x = nn.SELU(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.norm2(x)
        #x = self.dropout()
        x = self.fc2(x)
        
        return x
