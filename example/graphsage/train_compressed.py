import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.multiprocessing as mp
import dgl.nn.pytorch as dglnn
import time
import math
import argparse
from torch.nn.parallel import DistributedDataParallel
import tqdm
from model import SAGE
from utils.load_graph import *
import os
from torch.optim.lr_scheduler import ExponentialLR
from utils.compresser import Compresser

os.environ["OMP_NUM_THREADS"] = str(16)
th.multiprocessing.set_sharing_strategy('file_system')
class intsampler(dgl.dataloading.NeighborSampler):
    def __init__(self, fanouts, dev_id, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(fanouts, edge_dir, prob, replace,
                 prefetch_node_feats, prefetch_labels, prefetch_edge_feats,
                 output_device)  
        self.dev_id = dev_id

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # input_nodes, output_nodes, blocks
        src, dst, blocks = super().sample_blocks(g, seed_nodes, exclude_eids)
        blocks = [block.int() for block in blocks]
        return src, dst, blocks


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, nfeat, labels, val_nid, device, compresser):
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
        pred = th.zeros(g.num_nodes(), model.n_classes)
        sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            val_nid,
            sampler,
            use_ddp=False,
            device=device,
            batch_size=50,
            shuffle=False,
            drop_last=False,
            num_workers=0)
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            blocks = [block.to(device) for block in blocks]
            batch_inputs, batch_labels = load_subtensor(nfeat, labels,
                                                        output_nodes, input_nodes, device, compresser)                                                            
            result = model(blocks, batch_inputs).cpu()
            pred[output_nodes] = result
    model.train()

    return compute_acc(pred[val_nid], labels[val_nid])

def load_subtensor(nfeat, labels, seeds, input_nodes, dev_id, compresser):
    """
    Extracts features and labels for a subset of nodes.
    """
    exp = th.index_select(nfeat, 0, input_nodes.to(nfeat.device))
    exp = exp.to(dev_id, non_blocking=True)       
    batch_labels = th.index_select(labels, 0, seeds.to(labels.device))    
    batch_labels = batch_labels.to(dev_id, non_blocking=True)  
    ######### feature decompression #################################################### 
    batch_inputs = compresser.decompress(exp, dev_id)
    ####################################################################################
    return batch_inputs, batch_labels

#### Entry point

def run(proc_id, n_gpus, args, devices, data, compresser):
    # Start up distributed training, if enabled.
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    n_classes, train_g, val_g, test_g = data

    if args.inductive:
        train_nfeat = train_g.ndata.pop('features')
        val_nfeat = val_g.ndata.pop('features')
        test_nfeat = test_g.ndata.pop('features')
        train_labels = train_g.ndata.pop('labels').long()
        val_labels = val_g.ndata.pop('labels').long()
        test_labels = test_g.ndata.pop('labels').long()
    else:
        train_nfeat = val_nfeat = test_nfeat = g.ndata.pop('features')
        train_labels = val_labels = test_labels = g.ndata.pop('labels').long()        

    if args.data_gpu:
        train_nfeat = train_nfeat.to(dev_id)
        
    train_labels = train_labels.to(dev_id)

    train_mask = train_g.ndata.pop('train_mask')
    val_mask = val_g.ndata.pop('val_mask')
    test_mask = test_g.ndata.pop('test_mask')
    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()
    print(len(train_nid), len(val_nid), len(test_nid))

    if args.sample_gpu:
        train_g = train_g.to(dev_id)
        train_nid = train_nid.to(dev_id)

    # Create Pyth DataLoader for constructing blocks
    sampler = intsampler(
        [int(fanout) for fanout in args.fan_out.split(',')], dev_id)    

    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        use_ddp=n_gpus > 1,
        device=dev_id if args.num_workers == 0 and args.sample_gpu else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0 if args.sample_gpu else args.num_workers,
        persistent_workers=not args.sample_gpu,
        )


    # Define model and optimizer
    model = SAGE(compresser.feat_dim, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout, MLP=(args.dataset=="mag240m"), bn=False)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ExponentialLR(optimizer, gamma=0.98)

    iter_tput = []

    for epoch in range(args.num_epochs):
        if n_gpus > 1:
            dataloader.set_epoch(epoch)
        model.train()    
        tic = time.time()
        tic_step = time.time()
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(train_nfeat, train_labels,
                                                        seeds, input_nodes, dev_id, compresser)
            if proc_id == 0 and step==0 and epoch==0:
                print(blocks)
            blocks = [block.int().to(dev_id, non_blocking=True) for block in blocks]
            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if proc_id == 0:
                iter_tput.append(len(seeds) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB'.format(
                    epoch, step, float(loss.item()), float(compute_acc(batch_pred, batch_labels)), np.mean(iter_tput[3:]), th.cuda.max_memory_allocated() / 1000000))
            tic_step = time.time()

        toc = time.time()
        if proc_id == 0:            
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch % args.eval_every == 0 and epoch != 0:   
                eval_acc = 0  
                test_acc = 0             
                if n_gpus == 1:
                    eval_acc = evaluate(
                        model, val_g, val_nfeat, val_labels, val_nid, devices[0], compresser)
                    test_acc = evaluate(
                        model, test_g, test_nfeat, test_labels, test_nid, devices[0], compresser)
                else:
                    eval_acc = evaluate(
                        model.module, val_g, val_nfeat, val_labels, val_nid, devices[0], compresser)
                    if args.dataset!="mag240m":
                        test_acc = evaluate(
                            model.module, test_g, test_nfeat, test_labels, test_nid, devices[0], compresser)                                                         

                print('\nEval Acc {:.4f}'.format(eval_acc))
                print('Test Acc: {:.4f}'.format(test_acc))               
        scheduler.step()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0',
                           help="Comma separated list of GPU device IDs.")
    argparser.add_argument('--dataset', type=str, default='reddit')
    argparser.add_argument('--num-epochs', type=int, default=51)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=10)    
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=15,
                           help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--inductive', action='store_true',
                           help="Inductive learning setting")                         
    argparser.add_argument('--sample-gpu', action='store_true',
                           help="Inductive learning setting")                              
    argparser.add_argument('--data-gpu', action='store_true',
                           help="By default the script puts all node features and labels "
                                "on GPU when using it to save time for data copy. This may "
                                "be undesired if they cannot fit in GPU memory at once. "
                                "This flag disables that.")
    argparser.add_argument('--mode', type=str, default='sq')
    argparser.add_argument('--width', type=int, default=1)
    argparser.add_argument('--length', type=int, default=1)                                
    args = argparser.parse_args()

    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

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
    else:
        raise Exception('unknown dataset')


    print(g, n_classes)

    ######### feature compression ######################################################
    compresser = Compresser(args.mode, args.length, args.width)
    g.ndata["features"] = compresser.compress(g.ndata.pop("features"), args.dataset)
    ####################################################################################

    accs = []

    g = g.formats("csc")  
    if args.inductive:
        train_g, val_g, test_g = inductive_split(g)
    else:
        train_g = val_g = test_g = g

    data = n_classes, train_g, val_g, test_g      

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data, compresser)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data, compresser))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()




