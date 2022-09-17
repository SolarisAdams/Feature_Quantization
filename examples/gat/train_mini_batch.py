"""
Graph Attention Networks in DGL using SPMV optimization.
Multiple heads are also batched together for faster training.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import argparse
import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gat import GAT
from torch.optim.lr_scheduler import ExponentialLR
from utils.load_graph import *
from utils.compresser import Compresser

import tqdm
import os

os.environ["OMP_NUM_THREADS"] = str(16)
th.multiprocessing.set_sharing_strategy('file_system')
class intsampler(dgl.dataloading.MultiLayerNeighborSampler):
    def __init__(self, fanouts, dev_id=0, replace=False, return_eids=False):
        super().__init__(fanouts, replace, return_eids)  
        self.dev_id = dev_id

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # input_nodes, output_nodes, blocks
        blocks = super().sample_blocks(g, seed_nodes, exclude_eids)
        blocks = [block.int() for block in blocks]
        return blocks

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    indices = indices.to(labels.device)

    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def evaluate(model, g, nfeat, labels, val_nid, device, compresser, n_classes):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : A node ID tensor indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """

    model.eval()
    with th.no_grad():
        perm = th.randperm(len(val_nid))
        val_nid = val_nid[perm][:20000]
        pred = th.zeros(g.num_nodes(), n_classes)
        sampler = intsampler([int(30) for fanout in args.fan_out.split(',')], device)
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            val_nid,
            sampler,
            use_ddp=False,
            # device=device,
            device=None,
            batch_size=30,
            shuffle=False,
            drop_last=False,
            num_workers=0)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader, mininterval=1):
            # print(blocks)
            blocks = [block.to(device) for block in blocks]

            batch_inputs, batch_labels = load_subtensor(nfeat, labels,
                                                            output_nodes, input_nodes, device, compresser)                                                            
            result = model(blocks, batch_inputs).cpu()
            pred[output_nodes] = result
    model.train()
    return accuracy(pred[val_nid], labels[val_nid])


def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id, compresser):
    """
    Extracts features and labels for a subset of nodes.
    """
    # t1 = time.time()
    batch_inputs = th.index_select(nfeat, 0, input_nodes.to(nfeat.device))
    # t2 = time.time()

    batch_inputs = batch_inputs.to(dev_id, non_blocking=True)

    batch_labels = th.index_select(labels, 0, seeds.to(labels.device))
    batch_labels = batch_labels.to(dev_id, non_blocking=True)

    # t3 = time.time()

    batch_inputs = compresser.decompress(batch_inputs, dev_id)
    # t4 = time.time()
    # print('load_subtensor: ', t2-t1, t3-t2, t4-t3)
    return batch_inputs, batch_labels

def main(args):
    # load and preprocess dataset
    if args.dataset == 'reddit':
        g, n_classes = load_reddit()

    elif args.dataset == 'citeseer':
        g, n_classes = load_citeseer()

    elif args.dataset == 'cora':
        g, n_classes = load_cora()

    elif args.dataset == 'amazon':
        g, n_classes = load_amazon()  

    elif args.dataset == 'ogbn-products':
        g, n_classes = load_ogb('ogbn-products', root="/data/graphData/original_dataset")
    elif args.dataset == 'ogbn-papers100m':
        g, n_classes = load_ogbn_papers100m_in_subgraph()       
    elif args.dataset == 'mag240m':
        g, n_classes = load_mag240m_in_subgraph()         
    else:
        raise Exception('unknown dataset')

    print(g, n_classes)

####################################################################################################################
    compresser = Compresser(args.mode, args.length, args.width)
    g.ndata["features"] = compresser.compress(g.ndata.pop("features"), args.dataset)

    print(g.ndata['features'][:3])
####################################################################################################################

    feat_dim = compresser.feat_dim
    features = g.ndata.pop('features')
    labels = g.ndata.pop('labels').long()
    # labels = g.ndata['labels'].long()
    train_nid = g.ndata.pop('train_mask').nonzero().squeeze()
    val_nid = g.ndata.pop('val_mask').nonzero().squeeze()
    test_nid = g.ndata.pop('test_mask').nonzero().squeeze()
    # num_feats = features.shape[1]

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        if args.data_gpu:
            features = features.to(args.gpu)
            labels = labels.to(args.gpu)
        else:
            features = features.pin_memory()
            labels = labels.to(args.gpu)          
        # g = g.int().to(args.gpu)

    n_edges = g.number_of_edges()


    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           len(train_nid),
           len(val_nid),
           len(test_nid)))

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    # g = g.int()
    g = g.formats("csc")    
    n_edges = g.number_of_edges()


    n_gpus = 1
    # Create Pyth DataLoader for constructing blocks  
    # sampler = dgl.dataloading.MultiLayerNeighborSampler(
    #     [int(fanout) for fanout in args.fan_out.split(',')])
    sampler = intsampler(
        [int(fanout) for fanout in args.fan_out.split(',')], dev_id=args.gpu)
    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        use_ddp=n_gpus > 1,
        # device=args.gpu,
        device=args.gpu if args.num_workers == 0 and args.sample_gpu else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
        
        )


    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(args.num_layers-1,
                feat_dim,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    print(model)
    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.99)




    with open("results/loss_acc_"+str(args.gpu)+".txt", "a+") as f:
        print("============\nGAT", args.mode, args.length, args.width, args.dataset, args.fan_out, args.num_hidden, '\n', args, sep="\t", file=f)    


    tputs = 0
    acc = 0
    best_acc = 0
    avg_loss = 2
    # initialize graph
    dur = []
    time_list = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 1:
            t0 = time.time()
        t1 = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):

            t2 = time.time()
            batch_inputs, batch_labels = load_subtensor(features, labels, seeds, input_nodes, args.gpu, compresser)               
            # forward
            t23 = time.time()
            blocks = [block.to(args.gpu) for block in blocks]
            t3 = time.time()

            logits = model(blocks, batch_inputs)
            t4 = time.time()

            logp = F.log_softmax(logits, 1)
            loss = loss_fcn(logp, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t5 = time.time()
            avg_loss = 0.02*loss.detach() + 0.98*avg_loss
            if args.early_stop:
                if stopper.step(loss.item()):
                    print("Early stopping")
                    break
            tputs = 0.02*len(seeds) / (time.time() - t1)+0.98*tputs
            if step % args.log_every == 0:
                acc = 0.5*accuracy(logits, batch_labels)+0.5*acc
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Tputs {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB".
                        format(epoch, step, avg_loss.item(), tputs, acc, torch.cuda.max_memory_allocated() / 1024 ** 2))
            # print(time.time()-t1, t2-t1, t3-t2, t23-t2, t3-t23, t4-t3, t5-t4, time.time()-t5)
            time_list.append([time.time()-t1, t2-t1, t23-t2, t3-t23, t4-t3, t5-t4, time.time()-t5])
            t1 = time.time()

        scheduler.step()
        if epoch >= 1:
            dur.append(time.time() - t0)
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f}\n".
              format(epoch, np.mean(dur), avg_loss.item()))            
        if epoch % args.eval_every == args.eval_every - 1:
            model.eval()
            with th.no_grad():
                if args.dataset == 'mag240m':
                    test_acc = evaluate(model, g, features, labels, val_nid, args.gpu, compresser, n_classes)
                else:
                    test_acc = evaluate(model, g, features, labels, test_nid, args.gpu, compresser, n_classes)

                print("Epoch {:05d} | test Acc {:.4f} \n".
                        format(epoch, test_acc))
                if test_acc > best_acc:
                    best_acc = test_acc
                with open("results/loss_acc_"+str(args.gpu)+".txt", "a+") as f:
                    print(epoch+1, "{:.5f}\t{:.5f}\t{:.5f}".format(avg_loss.item(), 0, test_acc), sep="\t", file=f)                  




    print()
    print("Best Accuracy {:.4f}".format(best_acc))
    with open("results/time_log.txt", "a+") as f:
        for i in np.mean(time_list[3:], axis=0):
            print("{:.5f}".format(i), sep="\t", end="\t", file=f)
        print(args.mode, args.length, args.width, args.dataset, args.num_workers, args.gpu, args.batch_size, "GAT", sep="\t", file=f)         

    with open("results/acc.txt", "a") as f:
        print(args.dataset, args.width, args.length, "GAT", acc, sep="\t", file=f)            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    # register_data_args(parser)
    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--gpu", type=int, default=3,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument('--fan-out', type=str, default='5,10,15')                        
    parser.add_argument("--num-hidden", type=int, default=32,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.2,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--data-gpu', action="store_true", default=False)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--sample-gpu', action="store_true", default=False)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='sq')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--length', type=int, default=1)                           
    args = parser.parse_args()
    print(args)

    main(args)
