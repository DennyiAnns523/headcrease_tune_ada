#!/usr/bin/env python
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from backbones.iresnet import iresnet50
from config.config_hc import config as cfg
from utils.dataset import FaceDatasetFolder

def dump_embeddings():
    # 1) Prepare device & model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
    # Load checkpoint (might be half-precision)
    ckpt_path = (
        "/media/arjun/New Volume/DL_UbuntuFolderSSD/"
        "AdaDLProject/AdaDistill/output/headcrease/"
        "stage 1/best_backbone_epoch94.pth"
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    # Force all params & buffers to float32
    model = model.float().eval()
    for n, p in model.named_parameters():  p.data = p.data.float()
    for n, b in model.named_buffers():     b.data = b.data.float()

    # 2) Dataset + DataLoader (no shuffle, large number_sample)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = FaceDatasetFolder(
        root_dir=cfg.data_path,
        local_rank=0,
        number_sample=int(1e12)   # effectively 'no limit'
    )
    ds.transform = transform

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=getattr(cfg, 'num_workers', 8),
        pin_memory=True
    )

    # 3) Compute & collect embeddings
    all_embs = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            # force float32 on input
            imgs = imgs.to(device=device, dtype=torch.float32)
            embs = model(imgs).cpu().numpy()
            all_embs.append(embs)
            all_labels.append(labels.numpy())

    # 4) Stack & save
    embs_arr  = np.vstack(all_embs)
    labs_arr  = np.concatenate(all_labels)

    out_dir = cfg.output
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embs_arr)
    np.save(os.path.join(out_dir, "labels.npy"), labs_arr)

    print(f"Saved embeddings ({embs_arr.shape}) → {out_dir}/embeddings.npy")
    print(f"Saved labels     ({labs_arr.shape}) → {out_dir}/labels.npy")

if __name__ == "__main__":
    dump_embeddings()
