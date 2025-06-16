#!/usr/bin/env python
"""
compute_metrics.py

Run FaceNet-style verification, TAR@FAR, and Rank-1 identification metrics
on a given checkpoint, while loading the verification bin only once.
"""
import os
import glob
import argparse
import contextlib

import numpy as np
import torch
import torch.cuda.amp as amp
from sklearn.metrics import roc_curve, auc as sklearn_auc

from backbones.iresnet import iresnet50
from config.config_hc import config as cfg
from eval.metrics import (
    load_feature_pairs,
    calculate_roc_dist,
    tar_at_far,
    rank1_accuracy
)

# Disable all autocast contexts globally
amp.autocast = lambda *args, **kwargs: contextlib.nullcontext()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute verification metrics on a given backbone checkpoint, loading data once."
    )
    parser.add_argument(
        "--resume_epoch", type=int,
        help="Epoch number whose checkpoint is at cfg.output/epoch_XX_backbone.pth"
    )
    parser.add_argument(
        "--ckpt_file", type=str,
        help="Full path to a specific backbone .pth file (overrides --resume_epoch)"
    )
    parser.add_argument(
        "--nrof_folds", type=int, default=10,
        help="Number of folds for K-fold accuracy evaluation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine checkpoint path
    if args.ckpt_file:
        ckpt = args.ckpt_file
    elif args.resume_epoch is not None:
        ckpt = os.path.join(
            cfg.output,
            f"epoch_{args.resume_epoch:02d}_backbone.pth"
        )
    else:
        candidates = glob.glob(os.path.join(cfg.output, "best_backbone*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoint specified and no best_backbone*.pth under {cfg.output}"
            )
        ckpt = sorted(candidates)[-1]

    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    print(f"==> Loading checkpoint: {ckpt}")

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.float().to(device).eval()

    # Load distances & labels once
    protocol = os.path.join(cfg.verification_path, "fv1_test.bin")
    distances, issame = load_feature_pairs(
        protocol, cfg.data_path, model, device
    )

    # 1) FaceNet-style ROC & AUC + mean accuracy
    fpr, tpr, _ = roc_curve(issame, -distances)
    roc_auc = sklearn_auc(fpr, tpr)
    thresholds = np.arange(0.0, 2.0, 0.0001)
    _, _, _, mean_acc = calculate_roc_dist(
        thresholds, distances, issame, args.nrof_folds
    )
    print("\n--- FaceNet-style verification ---")
    print(f"Verification Accuracy: {mean_acc:.4%}")
    print(f"ROC AUC: {roc_auc:.4f}\n")

    # 2) TAR @ FAR
    print("--- TAR @ FAR ---")
    for far in (1e-2, 1e-3, 1e-4):
        tar, thr = tar_at_far(distances, issame, far)
        print(f"TAR @ FAR={far:.0e}: {tar:.4f}  (Threshold: {thr:.4f})")
    print()

    # 3) Rank-1 Identification
    # Assumes embeddings.npy & labels.npy exist
    try:
        embs = np.load("embeddings.npy")
        labs = np.load("labels.npy")
        r1 = rank1_accuracy(embs, labs)
        print("--- Rank-1 Identification ---")
        print(f"Rank-1 Accuracy: {r1:.4%}")
    except FileNotFoundError:
        print("embeddings.npy or labels.npy not found, skipping Rank-1 metric.")


if __name__ == '__main__':
    main()
