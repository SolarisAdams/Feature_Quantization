#!/bin/bash


python cluster_gcn.py --gpu 0 --dataset reddit-self-loop --lr 1e-2 --weight-decay 0.0 --psize 1500 --batch-size 20 \
  --n-epochs 30 --n-hidden 128 --n-layers 1 --log-every 100 --use-pp --self-loop \
  --note self-loop-reddit-non-sym-ly3-pp-cluster-2-2-wd-5e-4 --dropout 0.2 --use-val --normalize

/home/Adama/Envs/DGL/bin/python /home/Adama/DGL/feature_compression/cluster_gcn/cluster_gcn.py  --gpu 0 \
  --dataset ogbn-papers100m --lr 5e-4 --weight-decay 0.0 --psize 10000 --batch-size 30 --n-epochs 100 \
  --n-hidden 256 --n-layers 1 --log-every 50 --self-loop   --note self-loop-reddit-non-sym-ly3-nopp-cluster-2-2-wd-5e-4 \
  --dropout 0.2 --use-val --normalize