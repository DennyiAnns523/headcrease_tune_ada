# AdaDistill HeadCrease

## Paper

**AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition**
ECCV 2024 • [ArXiv](https://arxiv.org/abs/2407.01332)

## Problem

Recognize individuals by forehead crease patterns when masks cover most of the face.

## Installation

```bash
git clone https://github.com/DennyiAnns523/Adafinetune.git
cd Adafinetune
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data

* Place your HeadCrease dataset as:

  * `full_augmented/` (train)
  * `no_augmentation/` (val)
* Format: class‑labeled image folders
* Contact maintainer for access

## Usage

```bash
# 1. Train the base DeepFace teacher on MS1Mv3 data
bash run_standalone.sh

# 2. Fine‑tune the teacher on the HeadCrease dataset
bash run_fine_tune_teacher.sh
```

## Results

Fine‑tuned teacher achieves \~97.7% verification accuracy and AUC >0.99 on the HeadCrease validation set.

## Citation

```bibtex
@inproceedings{Boutros2024AdaDistill,
  title={AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  author={Boutros, Fadi and Štruc, Vitomir and Damer, Naser},
  booktitle={ECCV},
  year={2024}
}
```

