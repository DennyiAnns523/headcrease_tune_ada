#!/bin/bash

# Navigate to project root
cd /home/arjun/Downloads/AdaDLProject/AdaDistill

# Environment Setup
export OMP_NUM_THREADS=4
export PYTHONPATH=$(pwd):$PYTHONPATH

# NCCL Configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1  # Enable GPUs 0 and 1

# Launch distributed training using torch.distributed.launch
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=1237 \
  train/train_AdaDistill.py
