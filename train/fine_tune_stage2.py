#!/usr/bin/env python
import os
import torch
import numpy as np
import sklearn
from eval.verification_hc import load_bin, evaluate
from backbones.iresnet import iresnet50
from config.config_hc import config as cfg

def main():
    # 1. Prepare device and model (force float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE)
    # move to device **then** cast all params/buffers to float32
    model = model.to(device).float()


    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only layer3 and layer4
    for name, param in model.named_parameters():
        if name.startswith("layer3") or name.startswith("layer4"):
            param.requires_grad = True
            print(f"Unfreezing {name}")

    # also ensure every parameter tensor is float32
    for name, p in model.named_parameters():
        p.data = p.data.float()
    for name, buf in model.named_buffers():
        buf.data = buf.data.float()
    
  
    model.eval()

    # 2. Load your checkpoint
    ckpt_path = (
        "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output/headcrease/stage 2/best_backbone_epoch18.pth"
    )
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded weights from: {ckpt_path}\n")

    # 3. Find all .bin protocols
    protocol_dir = cfg.verification_path
    bin_paths = sorted(
        os.path.join(protocol_dir, fname)
        for fname in os.listdir(protocol_dir)
        if fname.endswith('.bin')
    )
    if not bin_paths:
        raise RuntimeError(f"No .bin files found in {protocol_dir}")

    # 4. Loop over each protocol
    for bin_path in bin_paths:
        print(f"Protocol: {os.path.basename(bin_path)}")
        data_list, issame = load_bin(bin_path)  # default 112×112

        embeddings = []
        with torch.no_grad():
            for data in data_list:
                # 5a. Cast input to float32 on GPU
                data = data.to(device=device, dtype=torch.float32)

                # DEBUG: confirm dtypes
                if data.dtype != torch.float32:
                    print(">> Warning: input still not float32:", data.dtype)

                # 5b. Normalize
                data = data / 255.0
                data = (data - 0.5) / 0.5

                # 5c. Inference
                emb = model(data).cpu().numpy()
                embeddings.append(emb)

        # 6. Fuse & normalize embeddings
        fused = sklearn.preprocessing.normalize(embeddings[0] + embeddings[1])

        # 7. Evaluate all metrics
        tpr, fpr, accs, val, val_std, far = evaluate(
            fused, np.array(issame, dtype=bool),
            nrof_folds=10, pca=0
        )

        # 8. Print summary stats
        print(f"  Mean Accuracy       : {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"  VAL @ FAR=1e-3      : {val:.4f} ± {val_std:.4f}")
        print(f"  Observed FAR        : {far:.4f}")

        # 9. (Optional) Full ROC curve
        print("  ROC curve (threshold → TPR, FPR):")
        for thr, T, F in zip(np.arange(0, 4, 0.01), tpr, fpr):
            print(f"    {thr:4.2f} → TPR {T:.4f}, FPR {F:.4f}")
        print()

if __name__ == "__main__":
    main()
