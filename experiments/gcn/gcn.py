"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import math

norm = "right"
class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        # n_classes = 100
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation, norm=norm))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation, norm=norm))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, activation=None, norm=norm))
        self.dropout = nn.Dropout(p=dropout)
        print("GCN layers:", len(self.layers))
        for i, layer in enumerate(self.layers):
            print("layer", i, layer)
            print("std", layer.weight.data[0].std())
            print("ratio", layer.weight.data[0].std() * math.sqrt((layer._in_feats + layer._in_feats)/2))
            print(layer.weight.data[0])

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            # print(h)

            if i != 0:
                h = self.dropout(h)
            # print(h)
            print("before layer", i, h.norm(p=2, dim=1).mean(), h.mean())

            h = layer(self.g, h)
            print("after layer", i, h.norm(p=2, dim=1).mean(), h.mean())
        return h
