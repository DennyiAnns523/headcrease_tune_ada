# # prepare_headbandCrease.py

# import os
# import random
# import shutil
# import pickle
# from glob import glob
# from tqdm import tqdm

# # ─── PARAMETERS ────────────────────────────────────────────────────────────────
# ROOT_DIR       = "/home/arjun/Downloads/AdaDLProject/AdaDistill/data/full_augmented_train_fv1"
# OUT_BASE       = "/home/arjun/Downloads/AdaDLProject/AdaDistill/data/headbandCrease"
# TRAIN_DIR      = os.path.join(OUT_BASE, "train")
# VAL_DIR        = os.path.join(OUT_BASE, "val")
# VAL_RATIO      = 0.1          # 10% per class → ~430×0.1≈43 val images/class
# PAIRS_OUT      = os.path.join(OUT_BASE, "forehead_verification.bin")
# NUM_SAME_PAIRS = 6000         # adjust as you like
# NUM_DIFF_PAIRS = 6000         # equal number of different-class pairs
# SEED           = 42
# # ────────────────────────────────────────────────────────────────────────────────

# random.seed(SEED)

# # 1) Create train/val directories (mirroring class subfolders)
# for d in (TRAIN_DIR, VAL_DIR):
#     if os.path.exists(d):
#         shutil.rmtree(d)
#     os.makedirs(d)

# # 2) Split each class
# classes = sorted(os.listdir(ROOT_DIR))
# for cls in tqdm(classes, desc="Splitting classes"):
#     src_cls = os.path.join(ROOT_DIR, cls)
#     imgs    = glob(os.path.join(src_cls, "*.jpg"))
#     random.shuffle(imgs)
#     cut = int(len(imgs) * VAL_RATIO)
#     val_imgs   = imgs[:cut]
#     train_imgs = imgs[cut:]

#     # make class subdirs in train/ and val/
#     os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
#     os.makedirs(os.path.join(VAL_DIR,   cls), exist_ok=True)

#     # copy
#     for p in train_imgs:
#         shutil.copy(p, os.path.join(TRAIN_DIR, cls, os.path.basename(p)))
#     for p in val_imgs:
#         shutil.copy(p, os.path.join(VAL_DIR,   cls, os.path.basename(p)))

# # 3) Build verification pairs from VAL_DIR
# pairs   = []
# issame  = []

# # 3a) Same-class pairs
# for cls in classes:
#     cls_imgs = glob(os.path.join(VAL_DIR, cls, "*.jpg"))
#     if len(cls_imgs) < 2:
#         continue
#     for _ in range(NUM_SAME_PAIRS // len(classes)):
#         a, b = random.sample(cls_imgs, 2)
#         pairs.append((a, b))
#         issame.append(True)

# # 3b) Different-class pairs
# all_val = []
# for cls in classes:
#     all_val.extend(glob(os.path.join(VAL_DIR, cls, "*.jpg")))
# for _ in range(NUM_DIFF_PAIRS):
#     a = random.choice(all_val)
#     b = random.choice(all_val)
#     # ensure different class
#     while os.path.dirname(a) == os.path.dirname(b):
#         b = random.choice(all_val)
#     pairs.append((a, b))
#     issame.append(False)

# # 4) Save protocol
# with open(PAIRS_OUT, "wb") as f:
#     pickle.dump({"pairs": pairs, "issame": issame}, f)

# print(f"Train/val split done. Protocol saved to {PAIRS_OUT}")


#!/usr/bin/env python
import os
import random
import pickle
import argparse
from itertools import combinations

def create_protocol_bin(data_dir, out_bin):
    # 1) Gather images per class
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_imgs = {}
    for cls in classes:
        folder = os.path.join(data_dir, cls)
        imgs = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if len(imgs) >= 2:
            class_imgs[cls] = imgs

    # 2) Make all positive pairs
    pos_pairs = []
    for imgs in class_imgs.values():
        pos_pairs += list(combinations(imgs, 2))
    n_pos = len(pos_pairs)

    # 3) Sample equal number of negative pairs
    neg_pairs = set()
    cls_list = list(class_imgs.keys())
    while len(neg_pairs) < n_pos:
        c1, c2 = random.sample(cls_list, 2)
        img1 = random.choice(class_imgs[c1])
        img2 = random.choice(class_imgs[c2])
        neg_pairs.add((img1, img2))
    neg_pairs = list(neg_pairs)

    # 4) Combine & shuffle
    pairs     = pos_pairs + neg_pairs
    issame    = [True]*n_pos + [False]*n_pos
    combined  = list(zip(pairs, issame))
    random.shuffle(combined)
    pairs, issame = zip(*combined)

    # 5) Read raw bytes into bins
    bins = []
    for (img_a, img_b) in pairs:
        with open(img_a, "rb") as f: bins.append(f.read())
        with open(img_b, "rb") as f: bins.append(f.read())

    # 6) Dump protocol
    with open(out_bin, "wb") as f:
        pickle.dump((bins, list(issame)), f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Written {len(issame)} pairs ({len(bins)} images) → {out_bin}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Make verification .bin from folder")
    p.add_argument("--data_dir", required=True,
                   help="path to test folder with class subfolders")
    p.add_argument("--out_bin",  required=True,
                   help="output .bin protocol file")
    args = p.parse_args()
    create_protocol_bin(args.data_dir, args.out_bin)

'''how to run
python veri_data.py \
  --data_dir "/data/FV1_TRAIN_TEST_WITHOUT_AUGMENTATION/test" \
  --out_bin  "/data/headbandCrease/fv1_test.bin"

'''
