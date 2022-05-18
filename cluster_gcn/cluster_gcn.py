import argparse
import os
import time
import random

import numpy as np
import networkx as nx
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args

from modules import GraphSAGE
from sampler import ClusterIter
from utils import Logger, evaluate, save_log_dir, load_data, calc_f1
from utils2.load_graph import *
from utils2.compresser import Compresser
from torch.optim.lr_scheduler import ExponentialLR

def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

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
        g, n_classes = load_ogb('ogbn-papers100M', root="/data/graphData/original_dataset")
        srcs, dsts = g.all_edges()
        g.add_edges(dsts, srcs)         
    # elif args.dataset == 'mag240m':
    #     g, n_classes, feats = load_mag240m()            
    # elif args.dataset == 'ogbn-papers100m':
    #     g, n_classes = load_ogbn_papers100m_in_subgraph()       
    elif args.dataset == 'mag240m':
        g, n_classes = load_mag240m_in_subgraph()          
    else:
        raise Exception('unknown dataset')

    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['labels'].long()

    train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
    val_nid = np.nonzero(val_mask.data.numpy())[0].astype(np.int64)
    test_nid = np.nonzero(test_mask.data.numpy())[0].astype(np.int64)
    all_nid = torch.tensor(np.concatenate([train_nid, val_nid, test_nid]))
    print(len(all_nid))


    # Normalize features
    if args.normalize:
        g.ndata['features'] = g.ndata["features"].sub_(g.ndata["features"][:100000].mean(dim=0)).div_(g.ndata["features"][:100000].std(dim=0))


    if args.dataset=="mag240m":
        in_feats = 768
    else:
        in_feats = g.ndata['features'].shape[1]

    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
            (n_edges, n_classes,
            n_train_samples,
            n_val_samples,
            n_test_samples))
    # create GCN model
    if args.self_loop and not args.dataset.startswith('reddit'):
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        print("adding self-loop edges")


    # metis only support int64 graph
    g = g.long()

    print(g)
    cluster_iterator = ClusterIter(
        args.dataset, g, args.psize, args.batch_size, all_nid, use_pp=args.use_pp)

    print('labels shape:', g.ndata['labels'].shape)
    # print("features shape, ", g.ndata['features'].shape)
    # features = cluster_iterator.g.ndata.pop('features')

    name = args.dataset
    if args.use_pp:
        name += '_pp'
    compresser = Compresser(args.mode, args.length, args.width)
    # if args.dataset=="mag240m":
    #     features = compresser.compress(feats, name, batch_size=50000)
    # else:
    features = cluster_iterator.g.ndata.pop('features')
    features = compresser.compress(features, name)

    print(cluster_iterator.g)
    cluster_iterator.g.ndata['train_mask'] = cluster_iterator.g.ndata['train_mask'].bool()
    cluster_iterator.g.ndata['val_mask'] = cluster_iterator.g.ndata['val_mask'].bool()
    cluster_iterator.g.ndata['test_mask'] = cluster_iterator.g.ndata['test_mask'].bool()

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        # g = g.int().to(args.gpu)


    

    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.use_pp)

    print(model)

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # Loss function
    if multitask:
        print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss()
    else:
        print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=0.97)

    # dataloader = torch.utils.data.DataLoader(
    #     cluster_iterator,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=4,
    # )

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
        print("current memory after model before training",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1
    loss_mean = 0
    tic = time.time()
    f1_list = []
    time_list = []

    for epoch in range(args.n_epochs):
        for j, cluster in enumerate(cluster_iterator):
            # sync with upper level training graph
            # print(cluster)
            t1 = time.time()
            batch_inputs = compresser.decompress(features[cluster.ndata["_ID"]], torch.cuda.current_device())
            t2 = time.time()
            # print(t2-t1)
            if epoch == 0 and j == 0:
                print(cluster)
            if cuda:
                cluster = cluster.to(torch.cuda.current_device())
            model.train()
            # forward
            pred = model(cluster, batch_inputs)
            batch_labels = cluster.ndata['labels'].long()
            batch_train_mask = cluster.ndata['train_mask']
            loss = loss_f(pred[batch_train_mask],
                          batch_labels[batch_train_mask])
            train_f1_mic, _ = calc_f1(batch_labels[batch_train_mask].detach().cpu().numpy(), 
                pred[batch_train_mask].detach().cpu().numpy(), multitask)
            f1_list.append(train_f1_mic)
            f1_list = f1_list[-100:]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del batch_inputs
            # in PPI case, `log_every` is chosen to log one time per epoch. 
            # Choose your log freq dynamically when you want more info within one epoch
            loss_mean = loss_mean*0.99 + loss.item()*0.01
            if j % args.log_every == 0:
                print(f"epoch:{epoch}/{args.n_epochs}, Iteration {j}/"
                      f"{len(cluster_iterator)}:training loss", loss_mean, "train f1:", np.mean(f1_list))
        print("current memory:",
              torch.cuda.memory_allocated(device=pred.device) / 1024 / 1024, "time:", time.time() - tic, "\n")
        time_list.append(time.time() - tic)
        scheduler.step()
        print(f'average time {np.mean(time_list)}')

        # evaluate
        if epoch % args.val_every == 0:
            val_f1_mic, val_f1_mac, test_f1_mic, test_f1_mac, test_acc = evaluate(
                model, g, labels, multitask, cluster_iterator, features, compresser)
            print("Val F1-mic{:.4f}, Val F1-mac{:.4f}, Test F1-mic{:.4f}, Test F1-mac{:.4f}, Test acc{:.4f}". 
                format(val_f1_mic, val_f1_mac, test_f1_mic, test_f1_mac, test_acc))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))
        tic = time.time()

    end_time = time.time()
    print(f'training using time {end_time - start_time}')
    print(f'average time {np.mean(time_list)}')

    # test
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    val_f1_mic, val_f1_mac, test_f1_mic, test_f1_mac, test_acc = evaluate(
        model, g, labels, multitask, cluster_iterator, features, compresser)
    print("Val F1-mic{:.4f}, Val F1-mac{:.4f}, Test F1-mic{:.4f}, Test F1-mac{:.4f}, Test acc{:.4f}". 
        format(val_f1_mic, val_f1_mac, test_f1_mic, test_f1_mac, test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--log-every", type=int, default=30,
                        help="the frequency to save model")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=1500,
                        help="partition number")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--val-every", type=int, default=5,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use precomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")
    parser.add_argument('--mode', type=str, default='sq')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--length', type=int, default=1)   
    args = parser.parse_args()

    print(args)

    main(args)
