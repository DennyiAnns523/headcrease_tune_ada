#!/usr/bin/env python
"""
eval/verification_hc.py: Verification routines for forehead-crease recognition.
Mirrors the official verification.py structure, using config_hc settings.
"""
import os
import glob
import datetime
import pickle

import mxnet as mx
import numpy as np
import sklearn
import torch
from mxnet import ndarray as nd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from backbones.iresnet import iresnet50
from config.config_hc import config as cfg


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
    predict = dist < threshold
    tp = np.sum(predict & actual_issame)
    fp = np.sum(predict & ~actual_issame)
    tn = np.sum(~predict & ~actual_issame)
    fn = np.sum(~predict & actual_issame)
    tpr = tp/(tp+fn) if (tp+fn)>0 else 0.0
    fpr = fp/(fp+tn) if (fp+tn)>0 else 0.0
    acc = (tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10, pca=0):
    assert embeddings1.shape == embeddings2.shape
    n = embeddings1.shape[0]
    kf = LFold(n_splits=nrof_folds)

    tprs = np.zeros((nrof_folds, len(thresholds)))
    fprs = np.zeros((nrof_folds, len(thresholds)))
    accs = np.zeros(nrof_folds)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)

    for i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        if pca>0:
            all_emb = np.vstack([embeddings1, embeddings2])
            pca_model = PCA(n_components=pca)
            pca_model.fit(all_emb)
            emb1 = sklearn.preprocessing.normalize(
                pca_model.transform(embeddings1))
            emb2 = sklearn.preprocessing.normalize(
                pca_model.transform(embeddings2))
            dist = np.sum((emb1-emb2)**2, axis=1)

        # train threshold
        acc_train = [calculate_accuracy(t, dist[train_idx], actual_issame[train_idx])[2]
                     for t in thresholds]
        best_thr = thresholds[np.argmax(acc_train)]

        for j, t in enumerate(thresholds):
            tprs[i,j], fprs[i,j], _ = calculate_accuracy(
                t, dist[test_idx], actual_issame[test_idx])
        _, _, accs[i] = calculate_accuracy(
            best_thr, dist[test_idx], actual_issame[test_idx])

    return np.mean(tprs,axis=0), np.mean(fprs,axis=0), accs


def calculate_val_far(threshold, dist, actual_issame):
    predict = dist < threshold
    ta = np.sum(predict & actual_issame)
    fa = np.sum(predict & ~actual_issame)
    n_same = np.sum(actual_issame)
    n_diff = np.sum(~actual_issame)
    val = ta/n_same if n_same>0 else 0.0
    far = fa/n_diff if n_diff>0 else 0.0
    return val, far


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame,
                  far_target, nrof_folds=10):
    assert embeddings1.shape == embeddings2.shape
    n = embeddings1.shape[0]
    kf = LFold(n_splits=nrof_folds)

    vals = np.zeros(nrof_folds)
    fars = np.zeros(nrof_folds)

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)

    for i, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        far_train = np.array([
            calculate_val_far(t, dist[train_idx], actual_issame[train_idx])[1]
            for t in thresholds])
        if far_train.max() >= far_target:
            thr = np.interp(far_target, far_train, thresholds)
        else:
            thr = 0.0
        vals[i], fars[i] = calculate_val_far(
            thr, dist[test_idx], actual_issame[test_idx])

    return vals.mean(), vals.std(), fars.mean()


def evaluate(embeddings, actual_issame, nrof_folds=10, pca=0):
    actual_issame = np.array(actual_issame, dtype=bool)
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    thr1 = np.arange(0,4,0.01)
    tpr, fpr, accs = calculate_roc(thr1, emb1, emb2, actual_issame, nrof_folds, pca)
    thr2 = np.arange(0,4,0.001)
    val, val_std, far = calculate_val(thr2, emb1, emb2, actual_issame, 1e-3, nrof_folds)
    return tpr, fpr, accs, val, val_std, far


@torch.no_grad()
def load_bin(path, image_size=(112,112)):
    """
    Load a .bin protocol and return two flips of tensors plus labels.
    """
    with open(path,'rb') as f:
        bins, issame_list = pickle.load(f)

    N = len(issame_list) * 2
    data_list = [torch.empty((N, 3, image_size[0], image_size[1])) for _ in range(2)]

    for idx in range(N):
        bin_data = bins[idx]
        img = mx.image.imdecode(bin_data)
        # resize to exact H×W
        img = mx.image.imresize(img, image_size[0], image_size[1])
        img = nd.transpose(img, (2, 0, 1))

        for flip in (0, 1):
            tensor = mx.ndarray.flip(img, axis=2) if flip else img
            data_list[flip][idx] = torch.from_numpy(tensor.asnumpy())

        if idx % 1000 == 0:
            print(f"Loading bin {idx}")

    issame_array = np.array(issame_list, dtype=bool)
    return data_list, issame_array


@torch.no_grad()
def test(data_set, backbone, batch_size, nfolds=10):
    data_list, issame_list = data_set
    embeddings_list = []

    for data in data_list:
        emb = None
        idx = 0
        while idx < data.shape[0]:
            end = min(idx + batch_size, data.shape[0])
            batch = data[end - batch_size:end]
            inp = ((batch / 255.0) - 0.5) / 0.5
            out = backbone(inp.cuda())
            arr = out.detach().cpu().numpy()
            if emb is None:
                emb = np.zeros((data.shape[0], arr.shape[1]))
            cnt = end - idx
            emb[idx:end, :] = arr[-cnt:]
            idx = end

        embeddings_list.append(emb)

    fused = sklearn.preprocessing.normalize(embeddings_list[0] + embeddings_list[1])
    _, _, accs, val, val_std, far = evaluate(fused, issame_list, nfolds)
    return 0.0, 0.0, float(accs.mean()), float(accs.std()), fused, embeddings_list


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
    ckpt = torch.load(os.path.join(cfg.output, 'best_backbone_epoch15.pth'), map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    bin_files = sorted(glob.glob(os.path.join(cfg.verification_path, '*.bin')))
    for path in bin_files:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"Evaluating protocol: {name}")
        start = datetime.datetime.now()
        data_set = load_bin(path)
        _, _, acc, acc_std, _, _ = test(data_set, model, cfg.batch_size)
        elapsed = (datetime.datetime.now() - start).total_seconds()
        print(f"  Accuracy: {acc*100:.2f}% ± {acc_std*100:.2f}% (Time: {elapsed:.1f}s)")

if __name__ == '__main__':
    main()
