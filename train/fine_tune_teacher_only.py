#!/usr/bin/env python
import os
import torch
import numpy as np
import sklearn
from eval.verification_hc import load_bin, evaluate
from backbones.iresnet import iresnet50
from config.config_hc import config as cfg

def main():
    # 1) Build model **on CPU**, in full-precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE)
    
    # 2) Freeze everything, then unfreeze just layer3, layer4, and head
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4") \
        or "arcface" in name or "classifier" in name:
            p.requires_grad = True
            print(f"Unfreezing {name}")
    
    # Ensure we start in full FP32
    model.float()
    
    # 3) Load your Stage-2 backbone checkpoint **onto CPU**, up-cast every tensor
    ckpt_path = (
        "/media/arjun/New Volume/DL_UbuntuFolderSSD/"
        "AdaDLProject/AdaDistill/output/headcrease/"
        "stage 2/best_backbone_epoch18.pth"
    )
    raw_ckpt = torch.load(ckpt_path, map_location="cpu")
    float_ckpt = {k: v.float() for k, v in raw_ckpt.items()}
    model.load_state_dict(float_ckpt, strict=False)
    print(f"Loaded and up-cast weights from: {ckpt_path}\n")
    
    # 4) Now move the fully-FP32 model to GPU and switch to eval
    model = model.to(device).eval()
    
    # 5) Locate all .bin protocols
    protocol_dir = cfg.verification_path
    bin_paths = sorted(
        os.path.join(protocol_dir, f)
        for f in os.listdir(protocol_dir)
        if f.endswith(".bin")
    )
    if not bin_paths:
        raise RuntimeError(f"No .bin files found in {protocol_dir}")
    
    # 6) Inference loop
    for bin_path in bin_paths:
        print(f"Protocol: {os.path.basename(bin_path)}")
        data_list, issame = load_bin(bin_path)  # default 112×112
        
        embeddings = []
        with torch.no_grad():
            for data in data_list:
                data = data.to(device, dtype=torch.float32)
                data = (data / 255.0 - 0.5) / 0.5
                emb = model(data).cpu().numpy()
                embeddings.append(emb)
        
        fused = sklearn.preprocessing.normalize(embeddings[0] + embeddings[1])
        tpr, fpr, accs, val, val_std, far = evaluate(
            fused, np.array(issame, dtype=bool),
            nrof_folds=10, pca=0
        )
        
        print(f"  Mean Accuracy       : {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"  VAL @ FAR=1e-3      : {val:.4f} ± {val_std:.4f}")
        print(f"  Observed FAR        : {far:.4f}\n")
        
        print("  ROC curve (threshold → TPR, FPR):")
        for thr, t, f in zip(np.arange(0, 4, 0.01), tpr, fpr):
            print(f"    {thr:4.2f} → TPR {t:.4f}, FPR {f:.4f}")
        print()

if __name__ == "__main__":
    main()
