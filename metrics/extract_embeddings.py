#!/usr/bin/env python
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from backbones.iresnet import iresnet50
from utils.dataset import FaceDatasetFolder
from config.config_hc import config as cfg

def main():
    # 1) Load fine-tuned backbone
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE)
    ckpt_path = "/media/arjun/New Volume1/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output/best_backbone_epoch15.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval().cuda()

    # 2) Prepare dataset & loader (no heavy augmentations)
    ds = FaceDatasetFolder(root_dir=cfg.data_path, local_rank=0)
    loader = DataLoader(
        ds,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    all_embs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            # Move to GPU
            imgs = imgs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Amp autocast to handle fp16/32
            with autocast():
                feats = model(imgs)  # (B, embedding_size)
            all_embs.append(feats.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 3) Save concatenated embeddings and labels
    embs = np.vstack(all_embs)    # shape (N, D)
    labs = np.concatenate(all_labels)  # shape (N,)
    np.save("embeddings.npy", embs)
    np.save("labels.npy", labs)
    print(f"Saved embeddings.npy ({embs.shape}) and labels.npy ({labs.shape})")

if __name__ == '__main__':
    main()
