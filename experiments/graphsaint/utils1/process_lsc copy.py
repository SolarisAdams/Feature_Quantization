import ogb
from ogb.lsc import MAG240MDataset
import tqdm
import numpy as np
import torch
import dgl
import dgl.function as fn
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--rootdir', type=str, default='.', help='Directory to download the OGB dataset.')
parser.add_argument('--author-output-path', type=str, help='Path to store the author features.')
parser.add_argument('--inst-output-path', type=str,
                    help='Path to store the institution features.')
parser.add_argument('--graph-output-path', type=str, help='Path to store the graph.')
parser.add_argument('--graph-format', type=str, default='csc', help='Graph format (coo, csr or csc).')
parser.add_argument('--graph-as-homogeneous', action='store_true', help='Store the graph as DGL homogeneous graph.')
parser.add_argument('--full-output-path', type=str,
                    help='Path to store features of all nodes.  Effective only when graph is homogeneous.')
args = parser.parse_args()

print('Building graph')
dataset = MAG240MDataset(root=args.rootdir)
ei_writes = dataset.edge_index('author', 'writes', 'paper')
ei_cites = dataset.edge_index('paper', 'paper')
ei_affiliated = dataset.edge_index('author', 'institution')

# We sort the nodes starting with the papers, then the authors, then the institutions.


g = dgl.heterograph({
    ('author', 'write', 'paper'): (ei_writes[0], ei_writes[1]),
    ('paper', 'write-by', 'author'): (ei_writes[1], ei_writes[0]),
    ('author', 'affiliate-with', 'institution'): (ei_affiliated[0], ei_affiliated[1]),
    ('institution', 'affiliate', 'author'): (ei_affiliated[1], ei_affiliated[0]),
    ('paper', 'cite', 'paper'): (np.concatenate([ei_cites[0], ei_cites[1]]), np.concatenate([ei_cites[1], ei_cites[0]]))
    })


paper_feat = dataset.paper_feat

if True:
    author_feat = torch.load(args.author_output_path)
    inst_feat = torch.load(args.inst_output_path)
else:
    author_feat = torch.empty((dataset.num_authors, dataset.num_paper_features), dtype=torch.float16)
    inst_feat = torch.empty((dataset.num_institutions, dataset.num_paper_features), dtype=torch.float16)
    # author_feat = np.memmap(args.author_output_path, mode='w+', dtype='float16', shape=(dataset.num_authors, dataset.num_paper_features))
    # inst_feat = np.memmap(args.inst_output_path, mode='w+', dtype='float16', shape=(dataset.num_institutions, dataset.num_paper_features))

    # Iteratively process author features along the feature dimension.
    BLOCK_COLS = 64
    with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
        for start in tq:
            tq.set_postfix_str('Reading paper features...')
            g.nodes['paper'].data['x'] = torch.FloatTensor(paper_feat[:, start:start + BLOCK_COLS].astype('float32'))
            # Compute author features...
            tq.set_postfix_str('Computing author features...')
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='write-by')
            # Then institution features...
            tq.set_postfix_str('Computing institution features...')
            g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliate-with')
            tq.set_postfix_str('Writing author features...')
            author_feat[:, start:start + BLOCK_COLS] = g.nodes['author'].data['x'].to(torch.float16)
            tq.set_postfix_str('Writing institution features...')
            inst_feat[:, start:start + BLOCK_COLS] = g.nodes['institution'].data['x'].to(torch.float16)
            del g.nodes['paper'].data['x']
            del g.nodes['author'].data['x']
            del g.nodes['institution'].data['x']
    torch.save(author_feat, args.author_output_path)
    torch.save(inst_feat, args.inst_output_path)


# paper_feat = torch.tensor(dataset.all_paper_feat, dtype=torch.float16)[g.nodes['paper'].data['_ID']]

# author_feat = torch.empty((g.number_of_nodes('author'), dataset.num_paper_features), dtype=torch.float16)
# inst_feat = torch.empty((g.number_of_nodes('institution'), dataset.num_paper_features), dtype=torch.float16)

# # Iteratively process author features along the feature dimension.
# BLOCK_COLS = 32
# with tqdm.trange(0, dataset.num_paper_features, BLOCK_COLS) as tq:
#     for start in tq:
#         tq.set_postfix_str('Reading paper features...')
#         g.nodes['paper'].data['x'] = paper_feat[:, start:start + BLOCK_COLS].to(torch.float32)
#         # Compute author features...
#         tq.set_postfix_str('Computing author features...')
#         g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='write-by')
#         # Then institution features...
#         tq.set_postfix_str('Computing institution features...')
#         g.update_all(fn.copy_u('x', 'm'), fn.mean('m', 'x'), etype='affiliate-with')
#         tq.set_postfix_str('Writing author features...')
#         author_feat[:, start:start + BLOCK_COLS] = g.nodes['author'].data['x'].to(torch.float16)
#         tq.set_postfix_str('Writing institution features...')
#         inst_feat[:, start:start + BLOCK_COLS] = g.nodes['institution'].data['x'].to(torch.float16)
#         del g.nodes['paper'].data['x']
#         del g.nodes['author'].data['x']
#         del g.nodes['institution'].data['x']
# torch.save(author_feat, args.author_output_path)
# torch.save(inst_feat, args.inst_output_path)

author_offset = 0
inst_offset = author_offset + g.number_of_nodes('author')
paper_offset = inst_offset + g.number_of_nodes('institution')

num_authors = g.number_of_nodes('author')
num_inst = g.number_of_nodes('institution')
num_papers = g.number_of_nodes('paper')

# Convert to homogeneous if needed.  (The RGAT baseline needs homogeneous graph)
if args.graph_as_homogeneous:
    # Process graph
    g = dgl.to_homogeneous(g)

    # DGL ensures that nodes with the same type are put together with the order preserved.
    # DGL also ensures that the node types are sorted in ascending order.
    assert torch.equal(
        g.ndata[dgl.NTYPE],
        torch.cat([torch.full((num_authors,), 0),
                   torch.full((num_inst,), 1),
                   torch.full((num_papers,), 2)]))
    assert torch.equal(
        g.ndata[dgl.NID],
        torch.cat([torch.arange(dataset.num_authors),
                   torch.arange(dataset.num_institutions),
                   torch.arange(dataset.num_papers)]))
    g.edata['etype'] = g.edata[dgl.ETYPE].byte()
    del g.edata[dgl.ETYPE]
    del g.ndata[dgl.NTYPE]
    del g.ndata[dgl.NID]




    # Process feature
    full_feat = np.memmap(
        args.full_output_path, mode='w+', dtype='float16',
        shape=(dataset.num_authors + dataset.num_institutions + dataset.num_papers, dataset.num_paper_features))
    BLOCK_ROWS = 100000
    for start in tqdm.trange(0, dataset.num_authors, BLOCK_ROWS):
        end = min(dataset.num_authors, start + BLOCK_ROWS)
        full_feat[author_offset + start:author_offset + end] = author_feat[start:end].numpy()
    for start in tqdm.trange(0, dataset.num_institutions, BLOCK_ROWS):
        end = min(dataset.num_institutions, start + BLOCK_ROWS)
        full_feat[inst_offset + start:inst_offset + end] = inst_feat[start:end].numpy()
    for start in tqdm.trange(0, dataset.num_papers, BLOCK_ROWS):
        end = min(dataset.num_papers, start + BLOCK_ROWS)
        full_feat[paper_offset + start:paper_offset + end] = paper_feat[start:end]
    
# Convert the graph to the given format and save.  (The RGAT baseline needs CSC graph)
g = g.formats(args.graph_format)
dgl.save_graphs(args.graph_output_path, g)