#!/bin/bash
cd /media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill
# Environment Setup
export OMP_NUM_THREADS=4
export PYTHONPATH=$(pwd):$PYTHONPATH

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SHM_DISABLE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1  # Enable GPUs 0 and 1

# Use torch.distributed.run (not launch)
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=1237 \
  train/train_standalone.py \
    --resume=1
