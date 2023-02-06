import torch.nn as nn
import dgl.nn as dglnn

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP=False):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, MLP):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.MLP = MLP        
        self.layers = nn.ModuleList()   
        method = 'mean'     
        if n_layers > 1:
            in_channel = n_hidden

            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, method))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(in_channel, n_hidden, method))

            if self.MLP:
                self.layers.append(dglnn.SAGEConv(in_channel, n_hidden, method))       
            else:         
                self.layers.append(dglnn.SAGEConv(in_channel, n_classes, method))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, method))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        if self.MLP:
            self.mlp = nn.Sequential(
                nn.Linear(n_hidden, n_hidden),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(n_hidden, n_classes),
            )            

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        if self.MLP:
            return self.mlp(h)
        else:
            return h
