
import dgl
import torch
from utils2.load_graph import *
dataset = 'mag240m'
# if dataset == 'ogbn-papers100m':
#     g, n_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
#     srcs, dsts = g.all_edges()
#     g.add_edges(dsts, srcs)         
if dataset == 'mag240m':
    g, n_classes, feats = load_mag240m()  

train_mask = g.ndata['train_mask']
val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
train_nid = train_mask.nonzero().squeeze()
val_nid = val_mask.nonzero().squeeze()
test_nid = test_mask.nonzero().squeeze()
all_nid = torch.cat([train_nid, val_nid, test_nid])
print(len(all_nid))

src, dst, eid = g.in_edges(all_nid, form='all')
l1_nid = src.unique()
print(len(l1_nid))
src, dst, eid = g.in_edges(l1_nid, form='all')
l2_nid = src.unique()
print(len(l2_nid))
# src, dst, eid = g.in_edges(l2_nid, form='all')
# l3_nid = src.unique()
subg = g.edge_subgraph(eid)
print(subg)
feats = torch.tensor(feats, dtype=torch.float16)
subg.ndata['features'] = feats[subg.ndata['_ID']]
subg.ndata.pop('_ID')
dgl.save_graphs("/data/giant_graph/in_subgraph/mag240m.dgl", [subg])
