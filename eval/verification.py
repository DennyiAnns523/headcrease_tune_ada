"""
verification.py: Verification routines for face recognition.
Generates ROC-style accuracy metrics on held-out pair protocols using MXNet and PyTorch.
"""

import datetime
import os
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from scipy import interpolate
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold


class LFold:
    """K-Fold splitter that falls back to full set if n_splits == 1."""
    def __init__(self, n_splits=10, shuffle=False):
        self.n_splits = n_splits
        if n_splits > 1:
            self.k_fold = KFold(n_splits=n_splits, shuffle=shuffle)

    def split(self, indices):
        if self.n_splits > 1:
            return self.k_fold.split(indices)
        return [(indices, indices)]


def calculate_accuracy(threshold, dist, actual_issame):
    """
    Compute TPR, FPR, and accuracy for a given distance threshold.

    Args:
        threshold (float): distance threshold.
        dist (ndarray): distances between embeddings.
        actual_issame (ndarray): boolean labels.

    Returns:
        (tpr, fpr, acc)
    """
    predict = dist < threshold
    tp = np.sum(predict & actual_issame)
    fp = np.sum(predict & ~actual_issame)
    tn = np.sum(~predict & ~actual_issame)
    fn = np.sum(~predict & actual_issame)

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    acc = (tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10, pca=0):
    """
    Perform ROC evaluation using K-fold cross-validation.

    Args:
        thresholds (ndarray): list of distance thresholds.
        embeddings1 (ndarray): first set of embeddings.
        embeddings2 (ndarray): second set of embeddings.
        actual_issame (ndarray): boolean labels.
        nrof_folds (int): number of folds.
        pca (int): PCA components (0 to disable).

    Returns:
        (mean_tpr, mean_fpr, accuracies_per_fold)
    """
    assert embeddings1.shape == embeddings2.shape
    n = embeddings1.shape[0]
    kf = LFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, len(thresholds)))
    fprs = np.zeros((nrof_folds, len(thresholds)))
    accs = np.zeros(nrof_folds)

    # initial squared L2 distances
    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)

    for i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        if pca > 0:
            all_emb = np.vstack([embeddings1, embeddings2])
            pca_model = PCA(n_components=pca)
            pca_model.fit(all_emb)
            e1 = pca_model.transform(embeddings1)
            e2 = pca_model.transform(embeddings2)
            e1 = sklearn.preprocessing.normalize(e1)
            e2 = sklearn.preprocessing.normalize(e2)
            dist = np.sum((e1 - e2) ** 2, axis=1)

        # find best threshold on train split
        acc_train = np.array([
            calculate_accuracy(t, dist[train_idx], actual_issame[train_idx])[2]
            for t in thresholds
        ])
        best_thr = thresholds[np.argmax(acc_train)]

        # record TPR/FPR on test split across all thresholds
        for j, t in enumerate(thresholds):
            tprs[i, j], fprs[i, j], _ = calculate_accuracy(
                t, dist[test_idx], actual_issame[test_idx]
            )

        # record accuracy at best threshold
        _, _, accs[i] = calculate_accuracy(
            best_thr, dist[test_idx], actual_issame[test_idx]
        )

    return np.mean(tprs, axis=0), np.mean(fprs, axis=0), accs


def calculate_val_far(threshold, dist, actual_issame):
    """
    Compute VAL and FAR at a given threshold.

    Returns:
        (val, far)
    """
    predict = dist < threshold
    ta = np.sum(predict & actual_issame)
    fa = np.sum(predict & ~actual_issame)
    n_same = np.sum(actual_issame)
    n_diff = np.sum(~actual_issame)
    val = ta / n_same if n_same > 0 else 0.0
    far = fa / n_diff if n_diff > 0 else 0.0
    return val, far


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame,
                  far_target, nrof_folds=10):
    """
    Find threshold for target FAR and compute VAL@FAR across folds.

    Returns:
        (mean_val, std_val, mean_far)
    """
    assert embeddings1.shape == embeddings2.shape
    n = embeddings1.shape[0]
    kf = LFold(n_splits=nrof_folds, shuffle=False)

    vals = np.zeros(nrof_folds)
    fars = np.zeros(nrof_folds)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)

    for i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        far_train = np.array([
            calculate_val_far(t, dist[train_idx], actual_issame[train_idx])[1]
            for t in thresholds
        ])
        if far_train.max() >= far_target:
            thr = interpolate.interp1d(far_train, thresholds)(far_target)
        else:
            thr = 0.0

        vals[i], fars[i] = calculate_val_far(
            thr, dist[test_idx], actual_issame[test_idx]
        )

    return vals.mean(), vals.std(), fars.mean()


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    """
    Convenience evaluator stub.

    Returns:
        (tpr, fpr, accuracies, val, val_std, far)
    """
    # ensure labels support NumPy indexing
    actual_issame = np.array(actual_issame, dtype=bool)

    # split even/odd embeddings
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]

    # ROC @ coarse thresholds
    thr1 = np.arange(0, 4, 0.01)
    tpr, fpr, accs = calculate_roc(thr1, emb1, emb2, actual_issame, nrof_folds, pca)

    # VAL@FAR @ finer thresholds
    thr2 = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thr2, emb1, emb2, actual_issame, 1e-3, nrof_folds)

    return tpr, fpr, accs, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size=(112, 112)):
    """
    Load a .bin protocol containing (bins, issame_list) and return
    two flips of image tensors plus labels as a NumPy array.

    Args:
        path (str):       path to the .bin file
        image_size (2-tuple): height and width to resize each image to

    Returns:
        data_list (list of 2 torch.Tensor of shape [2*N,3,H,W]),
        issame_array (np.ndarray of shape [N], dtype=bool)
    """
    with open(path, 'rb') as f:
        try:
            bins, issame_list = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            bins, issame_list = pickle.load(f, encoding='bytes')

    # Pre-allocate two flips per image
    N = len(issame_list) * 2
    data_list = [
        torch.empty((N, 3, image_size[0], image_size[1]))
        for _ in range(2)
    ]

    for idx in range(N):
        img = mx.image.imdecode(bins[idx])
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, (2, 0, 1))

        for flip in (0, 1):
            tensor = mx.ndarray.flip(img, axis=2) if flip else img
            data_list[flip][idx] = torch.from_numpy(tensor.asnumpy())

        if idx % 1000 == 0:
            print(f"Loading bin {idx}")

    # Convert labels list to a boolean NumPy array for correct indexing
    issame_array = np.array(issame_list, dtype=bool)
    return data_list, issame_array


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    """
    Run verification: compute embeddings on both flips, normalize, then evaluate.

    Returns:
        (acc1, std1, acc2, std2, xnorm, embeddings_list)
    """
    data_list, issame_list = data_set
    embeddings_list = []
    time_consumed = 0.0

    for data in data_list:
        emb = None
        idx = 0
        while idx < data.shape[0]:
            end = min(idx + batch_size, data.shape[0])
            batch = data[end - batch_size:end]
            t0 = datetime.datetime.now()
            inp = ((batch / 255.0) - 0.5) / 0.5
            out = backbone(inp)
            arr = out.detach().cpu().numpy()
            time_consumed += (datetime.datetime.now() - t0).total_seconds()

            if emb is None:
                emb = np.zeros((data.shape[0], arr.shape[1]))
            cnt = end - idx
            emb[idx:end, :] = arr[-cnt:]
            idx = end

        embeddings_list.append(emb)

    xnorm = np.linalg.norm(np.vstack(embeddings_list), axis=1).mean()
    fused = embeddings_list[0] + embeddings_list[1]
    fused = sklearn.preprocessing.normalize(fused)

    _, _, accs, val, val_std, far = evaluate(fused, issame_list, nfolds)
    return 0.0, 0.0, float(accs.mean()), float(accs.std()), xnorm, embeddings_list


def dumpR(data_set, backbone, batch_size, name='temp', data_extra=None):
    """
    Dump MXNet model embeddings to a .bin file.

    Args:
        data_set: (data_list, issame_list) from load_bin
        backbone: MXNet Module
        batch_size: int
        name: output filename prefix
        data_extra: optional extra data
    """
    _, issame_list = data_set
    embeddings_list = []

    for data in data_set[0]:
        emb = None
        idx = 0
        while idx < data.shape[0]:
            end = min(idx + batch_size, data.shape[0])
            batch = nd.slice_axis(data, axis=0, begin=end - batch_size, end=end)
            t0 = datetime.datetime.now()

            if data_extra is None:
                db = mx.io.DataBatch(data=(batch,), label=None)
            else:
                db = mx.io.DataBatch(data=(batch, data_extra), label=None)

            backbone.forward(db, is_train=False)
            out = backbone.get_outputs()[0].asnumpy()
            if emb is None:
                emb = np.zeros((data.shape[0], out.shape[1]))
            cnt = end - idx
            emb[idx:end, :] = out[-cnt:]
            idx = end

        embeddings_list.append(emb)

    fused = sklearn.preprocessing.normalize(embeddings_list[0] + embeddings_list[1])
    with open(f"{name}.bin", 'wb') as f:
        pickle.dump((fused, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
