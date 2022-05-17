#!/bin/bash


/home/Adama/Envs/DGL/bin/python /home/Adama/DGL/feature_compression/cluster_gcn/cluster_gcn.py --gpu 0 --dataset reddit --lr 1e-2 --weight-decay 0.0 --psize 1000 --batch-size 20 \
  --n-epochs 60 --n-hidden 128 --n-layers 1 --log-every 50 --use-pp --self-loop \
  --note self-loop-reddit-non-sym-ly3-pp-cluster-2-2-wd-5e-4 --dropout 0.25 --use-val --normalize

/home/Adama/Envs/DGL/bin/python /home/Adama/DGL/feature_compression/cluster_gcn/cluster_gcn.py  --gpu 0 \
  --dataset ogbn-papers100m --lr 1e-3 --weight-decay 0.0 --psize 2000 --batch-size 10 --n-epochs 100 \
  --n-hidden 256 --n-layers 1 --log-every 50 --self-loop   --note papers100m-non-sym-ly3-nopp-cluster-2-2-wd-5e-4 \
  --dropout 0.1 --use-val --normalize

/home/Adama/Envs/DGL/bin/python /home/Adama/DGL/feature_compression/cluster_gcn/cluster_gcn.py  --gpu 0 \
  --dataset mag240m --lr 1e-3 --weight-decay 0.0 --psize 4000 --batch-size 3 --n-epochs 100 \
  --n-hidden 256 --n-layers 1 --log-every 50 --note papers100m-non-sym-ly3-nopp-cluster-2-2-wd-5e-4 \
  --dropout 0.1 --use-val


  /home/Adama/Envs/DGL/bin/python /home/Adama/DGL/feature_compression/cluster_gcn/cluster_gcn.py  --gpu 2   --dataset mag240m --lr 1e-3 --weight-decay 0.0 --psize 4000 --batch-size 10 --n-epochs 100   --n-hidden 256 --n-layers 1 --log-every 50 --note papers100m-non-sym-ly3-nopp-cluster-2-2-wd-5e-4 --dropout 0.1 --use-val