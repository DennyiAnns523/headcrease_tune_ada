# AdaDistill HeadCrease

An adaptation of **AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition** applied to forehead crease recognition under masked conditions.

---

## 📖 Paper

**AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition**
ECCV 2024 • [arXiv:2407.01332](https://arxiv.org/abs/2407.01332)

---

## 🚚 Installation

```bash
git clone https://github.com/DennyiAnns523/Adafinetune.git
cd Adafinetune
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 📂 Data

Place your HeadCrease dataset with this structure:

```
full_augmented/    # Train images (augmented)
no_augmentation/    # Validation images
```

Each folder contains class-labeled subdirectories of images.

---

## 🚀 Usage

```bash
# 1. Train DeepFace Teacher on MS1Mv3 data
bash run_standalone.sh

# 2. Fine-tune Teacher on HeadCrease dataset
bash run_fine_tune_teacher.sh

```

---

## 📊 Example Results

```
Verification Accuracy: 97.69%
ROC AUC:             0.9971
TAR @ FAR=1e-2:      0.9613 
TAR @ FAR=1e-3:      0.9001 
TAR @ FAR=1e-4:      0.7998
```


---

## 📑 Citation

```bibtex
@inproceedings{Boutros2024AdaDistill,
  title={AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition},
  author={Boutros, Fadi and Štruc, Vitomir and Damer, Naser},
  booktitle={ECCV},
  year={2024}
}
```

---


