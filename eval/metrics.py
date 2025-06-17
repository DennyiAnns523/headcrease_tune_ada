import os
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc as sklearn_auc
import sklearn.preprocessing
import torch

from config.config_hc import config as cfg               # <-- import your batch_size
from eval.verification_hc import load_bin               # <-- HC loader that resizes

def load_feature_pairs(pairs_path: str,
                       root_dir: str,
                       model: torch.nn.Module,
                       device: str):
    """
    1) load_bin → two flips of every image as two big tensors + labels
    2) process each flip in mini-batches to avoid OOM
    3) fuse flips, compute pairwise distances
    """
    data_list, issame = load_bin(pairs_path)    # returns [flip0, flip1], issame_list
    model.eval()

    embs = []
    with torch.no_grad():
        for data in data_list:
            N = data.shape[0]
            all_out = []
            # process in chunks
            for start in range(0, N, cfg.batch_size):
                end = min(start + cfg.batch_size, N)
                batch = data[start:end].to(device)         # shape (B,3,112,112)
                # normalize [0,255]→[–1,1]
                batch = ((batch / 255.0) - 0.5) / 0.5
                out = model(batch)
                all_out.append(out.detach().cpu().numpy())
            embs.append(np.vstack(all_out))               # shape (N, feat_dim)

    # fuse flips and L2-normalize
    fused = sklearn.preprocessing.normalize(embs[0] + embs[1])

    # compute squared L2 distances for each pair
    emb1 = fused[0::2]
    emb2 = fused[1::2]
    distances = np.sum((emb1 - emb2)**2, axis=1)

    return distances, np.array(issame, dtype=bool)


def calculate_accuracy(threshold: float,
                       dist: np.ndarray,
                       actual_issame: np.ndarray):
    predict = dist < threshold
    tp = np.sum(predict & actual_issame)
    tn = np.sum(~predict & ~actual_issame)
    fp = np.sum(predict & ~actual_issame)
    fn = np.sum(~predict & actual_issame)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / dist.size
    return tpr, tnr, acc


def calculate_roc_dist(thresholds: np.ndarray,
                       dist: np.ndarray,
                       actual_issame: np.ndarray,
                       nrof_folds: int = 10):
    n = len(actual_issame)
    kf = KFold(n_splits=nrof_folds, shuffle=False)
    tprs, fprs, accs = [], [], []

    for train_idx, test_idx in kf.split(np.arange(n)):
        # find best threshold on train split
        best_acc, best_thr = 0.0, thresholds[0]
        for thr in thresholds:
            _, _, acc = calculate_accuracy(
                thr, dist[train_idx], actual_issame[train_idx]
            )
            if acc > best_acc:
                best_acc, best_thr = acc, thr

        # evaluate on test split
        tpr, tnr, acc = calculate_accuracy(
            best_thr, dist[test_idx], actual_issame[test_idx]
        )
        tprs.append(tpr)
        fprs.append(1.0 - tnr)
        accs.append(acc)

    mean_tpr = float(np.mean(tprs))
    mean_fpr = float(np.mean(fprs))
    mean_acc = float(np.mean(accs))
    auc_score = float(np.trapz(tprs, fprs))
    return mean_fpr, mean_tpr, auc_score, mean_acc


def tar_at_far(distances: np.ndarray,
               issame: np.ndarray,
               far_target: float = 1e-4):
    thresholds = np.arange(0.0, 2.0, 0.0001)
    tpr_list, fpr_list = [], []
    for thr in thresholds:
        tpr, tnr, _ = calculate_accuracy(thr, distances, issame)
        tpr_list.append(tpr)
        fpr_list.append(1.0 - tnr)

    tpr_arr = np.array(tpr_list)
    fpr_arr = np.array(fpr_list)
    if far_target > fpr_arr.max():
        idx = np.argmax(fpr_arr)
        return float(tpr_arr[idx]), float(thresholds[idx])
    tar = float(np.interp(far_target, fpr_arr, tpr_arr))
    thr = float(np.interp(far_target, fpr_arr, thresholds))
    return tar, thr


def rank1_accuracy(embeddings: np.ndarray,
                   labels: np.ndarray) -> float:
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    N = embeddings.shape[0]
    dists = np.sum(
        (embeddings[:, None, :] - embeddings[None, :, :]) ** 2,
        axis=2
    )
    correct = 0
    for i in range(N):
        dists[i, i] = np.inf
        nn = np.argmin(dists[i])
        if labels[nn] == labels[i]:
            correct += 1
    return float(correct) / float(N)


def evaluate_model(model: torch.nn.Module,
                   pairs_path: str,
                   root_dir: str,
                   device: str,
                   nrof_folds: int = 10) -> dict:
    distances, issame = load_feature_pairs(pairs_path, root_dir, model, device)

    # ROC & AUC
    scores = -distances
    fpr, tpr, _ = roc_curve(issame, scores)
    roc_auc = sklearn_auc(fpr, tpr)

    # mean accuracy via K-fold
    thresholds = np.arange(0.0, 2.0, 0.0001)
    _, _, _, mean_acc = calculate_roc_dist(
        thresholds, distances, issame, nrof_folds
    )

    return {
        'fpr': fpr,
        'tpr': tpr,
        'auc': float(roc_auc),
        'accuracy': float(mean_acc)
    }
