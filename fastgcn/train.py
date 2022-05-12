import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models import GCN
from sampler import Sampler_FastGCN, Sampler_ASGCN
from utils.utils import load_data, get_batches, accuracy, sparse_mx_to_torch_sparse_tensor
from utils.load_graph import *
from utils.compresser import Compresser
import scipy.sparse as sp


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora',
                        help='dataset name.')
    # model can be "Fast" or "AS"
    parser.add_argument('--model', type=str, default='Fast',
                        help='model name.')
    parser.add_argument('--test_gap', type=int, default=10,
                        help='the train epochs between two test')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0007,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=16,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--batchsize', type=int, default=256,
                        help='batchsize for train')
    parser.add_argument('--mode', type=str, default='sq')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--length', type=int, default=1)                             
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def train(train_ind, train_labels, batch_size, train_times):
    t = time.time()
    model.train()
    for epoch in range(train_times):
        for batch_inds, batch_labels in get_batches(train_ind,
                                                    train_labels,
                                                    batch_size):
            sampled_feats, sampled_adjs, var_loss = model.sampling(
                batch_inds)
            # print(sampled_feats.shape)
            optimizer.zero_grad()
            output = model(sampled_feats, sampled_adjs)
            loss_train = loss_fn(output, batch_labels) + 0.5 * var_loss
            acc_train = accuracy(output, batch_labels)
            loss_train.backward()
            optimizer.step()
    # just return the train loss of the last train epoch
    return loss_train.item(), acc_train.item(), time.time() - t


def test(test_adj, test_feats, test_labels, epoch):
    t = time.time()
    model.eval()
    outputs = model(test_feats, test_adj)
    loss_test = loss_fn(outputs, test_labels)
    acc_test = accuracy(outputs, test_labels)

    return loss_test.item(), acc_test.item(), time.time() - t


if __name__ == '__main__':
    # load data, set superpara and constant
    args = get_args()

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
        g, n_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)         
    elif args.dataset == 'mag240m':
        g, n_classes, feats = load_mag240m()            
    else:
        raise Exception('unknown dataset')
    print(g)

    # adj, features, adj_train, train_features, y_train, y_test, test_index = \
    #     load_data(args.dataset)
    # print(adj)
    # print(g.adj().coalesce().indices().numpy())

    adj = adj_train = sp.coo_matrix(((np.ones(g.num_edges()),g.adj().coalesce().indices().numpy()))).tocsr()

    ##########################################################################################################
    compresser = Compresser(args.mode, args.length, args.width)
    if args.dataset=="mag240m":
        
        g.ndata["features"] = compresser.compress(feats, batch_size=50000)
    else:
        g.ndata["features"] = compresser.compress(g.ndata.pop("features"))
    ##########################################################################################################



    features = train_features = g.ndata["features"]
    y_train = y_test = F.one_hot(g.ndata["labels"], num_classes=n_classes)
    test_index = torch.nonzero(g.ndata["test_mask"]).flatten()
    train_index = torch.nonzero(g.ndata["test_mask"]).flatten()
    del g
    y_test = y_test[test_index]
    # y_train = y_train[train_index]




    layer_sizes = [128, 128]
    input_dim = compresser.feat_dim
    train_nums = adj_train.shape[0]
    test_gap = args.test_gap
    nclass = y_train.shape[1]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # set device
    if args.cuda:
        device = torch.device(3)
        print("use cuda")
    else:
        device = torch.device("cpu")

    # data for train and test
    # features = torch.FloatTensor(features).to(device)
    # train_features = torch.FloatTensor(train_features).to(device)
    y_train = torch.LongTensor(y_train).to(device).max(1)[1]
    # print(test_index)
    # print(adj)
    test_adj = [adj, adj[test_index, :]]
    test_feats = features
    test_labels = y_test
    test_adj = [sparse_mx_to_torch_sparse_tensor(cur_adj).to(device)
                for cur_adj in test_adj]
    test_labels = torch.LongTensor(test_labels).to(device).max(1)[1]

    # init the sampler
    if args.model == 'Fast':
        sampler = Sampler_FastGCN(None, train_features, adj_train,
                                  input_dim=input_dim,
                                  layer_sizes=layer_sizes,
                                  device=device, compresser=compresser)
    elif args.model == 'AS':
        sampler = Sampler_ASGCN(None, train_features, adj_train,
                                input_dim=input_dim,
                                layer_sizes=layer_sizes,
                                device=device, compresser=compresser)
    else:
        print(f"model name error, no model named {args.model}")
        exit()

    # init model, optimizer and loss function
    model = GCN(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=nclass,
                dropout=args.dropout,
                sampler=sampler).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = F.nll_loss
    # loss_fn = torch.nn.CrossEntropyLoss()

    # train and test
    for epochs in range(0, args.epochs // test_gap):
        train_loss, train_acc, train_time = train(np.arange(train_nums),
                                                  y_train,
                                                  args.batchsize,
                                                  test_gap)
        test_loss, test_acc, test_time = test(test_adj,
                                              test_feats,
                                              test_labels,
                                              args.epochs)
        print(f"epchs:{epochs * test_gap}~{(epochs + 1) * test_gap - 1} "
              f"train_loss: {train_loss:.3f}, "
              f"train_acc: {train_acc:.3f}, "
              f"train_times: {train_time:.3f}s "
              f"test_loss: {test_loss:.3f}, "
              f"test_acc: {test_acc:.3f}, "
              f"test_times: {test_time:.3f}s")
    with open("results/acc.txt", "a") as f:
        print(args.dataset, args.width, args.lens, "FastGCN", test_acc, sep="\t", file=f)    
