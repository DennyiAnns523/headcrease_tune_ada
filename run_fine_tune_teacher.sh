# #!/bin/bash

# # Navigate to project root
# cd "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill"

# # Ensure a writable temp directory #for ptogtrssove finetune
# mkdir -p ~/tmp
# export TMPDIR=~/tmp

# # Environment Setup
# export OMP_NUM_THREADS=4
# export PYTHONPATH=$(pwd):$PYTHONPATH

# # NCCL Configuration
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0,1  # Enable GPUs 0 and 1

# # Launch distributed fine-tuning of the teacher
# python -m torch.distributed.launch \
#   --nproc_per_node=2 \
#   --nnodes=1 \
#   --node_rank=0 \
#   --master_addr=127.0.0.1 \
#   --master_port=1238 \
#   train/fine_tune_teacher_only.py
    

#!/bin/bash
cd "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill"
#!/bin/bash

# Go to project root
cd "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill"

# Environment
export OMP_NUM_THREADS=4
export PYTHONPATH=$(pwd):$PYTHONPATH
export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

# Launch Stage-3 fine-tuning
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --master_port=1238 \
  train/fine_tune_teacher_only.py \
    --backbone_ckpt "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output/headcrease/stage 2/best_backbone_epoch18.pth" \
    --header_ckpt   "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output/headcrease/stage 2/best_header_epoch18.pth"
