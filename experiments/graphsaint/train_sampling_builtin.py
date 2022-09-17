import argparse
import os
import time
import torch
import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from sampler import SAINTNodeSampler, SAINTEdgeSampler, SAINTRandomWalkSampler
from config import CONFIG
from modules import GCNNet
from utils import Logger, evaluate, save_log_dir, load_data, calc_f1
import warnings
from utils1.compresser import Compresser
from dgl.dataloading import SAINTSampler, DataLoader
from model import SAGE

def main(args, task):
    warnings.filterwarnings('ignore')
    multilabel_data = {'ppi', 'yelp', 'amazon'}
    multilabel = args.dataset in multilabel_data

    # This flag is excluded for too large dataset, like amazon, the graph of which is too large to be directly
    # shifted to one gpu. So we need to
    # 1. put the whole graph on cpu, and put the subgraphs on gpu in training phase
    # 2. put the model on gpu in training phase, and put the model on cpu in validation/testing phase
    # We need to judge cpu_flag and cuda (below) simultaneously when shift model between cpu and gpu
    if args.dataset in ['amazon']:
        cpu_flag = True
    else:
        cpu_flag = False
    cpu_flag = True

    # load and preprocess dataset
    g, n_classes = load_data(args, multilabel)
    compresser = Compresser(args.mode, args.length, args.width)
    if args.dataset=="mag240m":
        
        g.ndata["features"] = compresser.compress(feats, args.dataset, batch_size=50000)
    else:
        g.ndata["features"] = compresser.compress(g.ndata.pop("features"), args.dataset)

    # g = data.g
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    labels = g.ndata['labels']

    train_nid = g.ndata['train_mask'].nonzero().squeeze()

    in_feats = compresser.feat_dim
    n_nodes = g.num_nodes()
    n_edges = g.num_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Nodes %d
    #Edges %d
    #Classes/Labels (multi binary labels) %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_nodes, n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    # load sampler

    kwargs = {
        'dn': args.dataset, 'g': g, 'train_nid': train_nid, 'num_workers_sampler': args.num_workers_sampler,
        'num_subg_sampler': args.num_subg_sampler, 'batch_size_sampler': args.batch_size_sampler,
        'online': args.online, 'num_subg': args.num_subg, 'full': args.full
    }

    # if args.sampler == "node":
    #     saint_sampler = SAINTNodeSampler(args.node_budget, **kwargs)
    # elif args.sampler == "edge":
    #     saint_sampler = SAINTEdgeSampler(args.edge_budget, **kwargs)
    # elif args.sampler == "rw":
    #     saint_sampler = SAINTRandomWalkSampler(args.num_roots, args.rwlength, **kwargs)
    # else:
    #     raise NotImplementedError
    # loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1,
    #                     shuffle=True, num_workers=args.num_workers, drop_last=False)
    num_iters = 1000
    sampler = SAINTSampler(mode='walk', budget=(args.num_roots, args.rwlength))
    # Assume g.ndata['feat'] and g.ndata['label'] hold node features and labels
    loader = DataLoader(g, torch.arange(num_iters), sampler, num_workers=args.num_workers)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if not cpu_flag:
            g = g.to('cuda:{}'.format(args.gpu))

    print('labels shape:', g.ndata['labels'].shape)
    print("features shape:", g.ndata['features'].shape)

    # model = GCNNet(
    #     in_dim=in_feats,
    #     hid_dim=args.n_hidden,
    #     out_dim=n_classes,
    #     arch=args.arch,
    #     dropout=args.dropout,
    #     batch_norm=not args.no_batch_norm,
    #     aggr=args.aggr
    # )

    model = SAGE(compresser.feat_dim, args.n_hidden, n_classes, len(args.arch.split("-")), 
        F.relu, args.dropout, MLP=(args.dataset=="mag240m"), bn=False)

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = train_nid.cuda()
        print("GPU memory allocated before training(MB)",
              torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1
    tick = time.time()
    t0 = time.time()
    for epoch in range(args.n_epochs):
        for j, subg in enumerate(loader):
            t1 = time.time()
            if epoch == 0 and j <= 10:
                print(subg, subg.ndata['features'].shape)          
            if cuda:
                batch_inputs = compresser.decompress(subg.ndata.pop("features").to(torch.cuda.current_device()), torch.cuda.current_device())             

                subg = subg.to(torch.cuda.current_device())
            else:
                batch_inputs = compresser.decompress(subg.ndata.pop("features"), torch.cuda.current_device()) 
            t2 = time.time()
           
            model.train()
            # forward     
            pred = model(subg, batch_inputs)
            t3 = time.time()
            batch_labels = subg.ndata['labels']

            if multilabel:
                loss = F.binary_cross_entropy_with_logits(pred, batch_labels, reduction='sum',
                                                          weight=subg.ndata['l_n'].unsqueeze(1))
            else:
                loss = F.cross_entropy(pred, batch_labels, reduction='none')
                loss = loss.sum()
            t4 = time.time()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()
            t5 = time.time()
            if j % 100 == 0:
                print(j, t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)
            if j == len(loader)//args.num_subg*3 - 1:
                model.eval()
                with torch.no_grad():
                    train_f1_mic, train_f1_mac = calc_f1(batch_labels.cpu().numpy(),
                                                         pred.cpu().numpy(), multilabel)
                    print(f"epoch:{epoch + 1}/{args.n_epochs}, Iteration {j + 1}/"
                          f"{len(loader)}:training loss", loss.item())
                    print("Train F1-mic {:.4f}, Train F1-mac {:.4f}".format(train_f1_mic, train_f1_mac))
                    # log time
                    print("Epoch {}/{} took {:.3f}s".format(epoch + 1, args.n_epochs, time.time() - tick))
                    print(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4)
                break
            t0 = t5
            
                    
        # evaluate
        model.eval()
        if epoch % args.val_every == 0:
            # if cpu_flag and cuda:  # Only when we have shifted model to gpu and we need to shift it back on cpu
            #     model = model.to('cpu')
            val_f1_mic, val_f1_mac = evaluate(
                model, g, labels, val_mask, multilabel, compresser, cpu_flag)
            print(
                "Val F1-mic {:.4f}, Val F1-mac {:.4f}".format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model_{}.pkl'.format(task)))
            if cpu_flag and cuda:
                model.cuda()
        tick = time.time()

    end_time = time.time()
    print(f'training using time {end_time - start_time}')

    # test
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model_{}.pkl'.format(task))))
    if cpu_flag and cuda:
        model = model.to('cpu')
    test_f1_mic, test_f1_mac = evaluate(
        model, g, labels, test_mask, multilabel, compresser, cpu_flag)
    print("Test F1-mic {:.4f}, Test F1-mac {:.4f}".format(test_f1_mic, test_f1_mac))

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser(description='GraphSAINT')
    parser.add_argument("--task", type=str, default="default", help="type of tasks")
    parser.add_argument("--online", dest='online', action='store_true', help="sampling method in training phase")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument('--mode', type=str, default='sq')
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--length', type=int, default=1)     
    task = parser.parse_args().task
    args = argparse.Namespace(**CONFIG[task])
    args.online = parser.parse_args().online
    args.gpu = parser.parse_args().gpu
    args.mode = parser.parse_args().mode
    args.width = parser.parse_args().width
    args.length = parser.parse_args().length
    print(args)

    main(args, task=task)
