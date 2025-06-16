# from easydict import EasyDict as edict
# import os

# config = edict()

# # —— Dataset & I/O ——
# config.dataset = "headbandCrease"
# config.db_file_format = "folder"
# config.data_path = "/home/arjun/Downloads/AdaDLProject/AdaDistill/data/full_augmented_train_fv1"
# config.output    = "output/headbandCrease_ft/"

# # Base directory containing train/, val/, and .bin file
# config.base_dir =  "/home/arjun/Downloads/AdaDLProject/AdaDistill/data/headbandCrease"

# # Training folder
# config.data_path = os.path.join(config.base_dir, "train")
# # Where validation .bin lives
# config.verification_path = config.base_dir

# # —— Model & Loss ——
# config.network         = "iresnet50"    # backbone architecture
# config.teacher         = "iresnet50"    # teacher backbone
# config.embedding_size  = 512            # output embedding dim of the backbone
# config.SE              = True          # whether to use squeeze-and-excitation
# config.loss            = "ArcFace"      # classification loss
# config.s               = 80.0 #32.0           # ArcFace scale (↓ from 64.0 for safety)
# config.m               = 0.35           # ArcFace margin (↓ from 0.45 for stability)
# config.adaptive_alpha  = True           # AdaDistill alpha scheduling

# # —— Classes & Data stats ——
# config.num_classes     = 247
# config.num_image       = 106210         # approx. total images
# config.sample          = int(1e9)       # no hard cap on sample count

# # —— Fine-tuning hyperparameters ——
# config.batch_size      = 256         # total batch size across all GPUs
# config.ft_lr           = 0.05          # reduced LR for safety
# config.weight_decay    = 5e-4
# config.ft_epochs       = 100
# config.resume_epoch =  50 #may change later

# # —— Evaluation & logging ——
# config.eval_step       = int(config.num_image / config.batch_size + 0.5)
# config.val_targets     = ["forehead_verification"]

# # —— Learning-rate schedule ——
# # Warmup for 5 epochs, then decay by 0.1 at 30, 60, 80
# def lr_step_func(epoch):
#     if epoch < 5:
#         return (epoch + 1) / 5
#     return 0.1 ** len([m for m in [30, 60, 80] if epoch >= m])

# config.lr_func = lr_step_func

# # —— Teacher checkpoint ——
# config.pretrained_teacher_path = (
#     "/home/arjun/Downloads/AdaDLProject/AdaDistill/"
#     "output/teacher/147836header.pth"
# )

# # —— Misc ——
# config.momentum = 0.9


from easydict import EasyDict as edict
import os

config = edict()

# —— Dataset & I/O ——
config.dataset          = "FV1_TRAIN_TEST_WITHOUT_AUGMENTATION"  # updated dataset name
config.db_file_format   = "folder"
# Path to non-augmented FV1 train/test data
config.data_path        = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/data/FV1_TRAIN_TEST_WITHOUT_AUGMENTATION/train"
# Directory where .bin verification protocols reside
config.verification_path = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/data"
# Output directory for fine-tuned checkpoints and logs
config.output           = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output/headcrease/stage3"

# —— Model & Loss ——
config.network          = "iresnet50"               # backbone architecture
config.teacher          = "iresnet50"               # teacher backbone
config.embedding_size   = 512                        # embedding dimension
config.SE               = True                       # use squeeze-and-excitation
config.loss             = "ArcFace"                 # loss type
config.s                = 80.0                       # ArcFace scale
config.m                = 0.35                       # ArcFace margin
config.adaptive_alpha   = True                       # enable adaptive alpha scheduling

# —— Classes & Data Statistics ——
config.num_classes      = 247                        # number of classes in training
config.num_image        = 106210                     # total training images
config.sample           = int(1e9)                   # no sample cap

# —— Fine-Tuning Hyperparameters ——
config.batch_size       = 256                        # total batch size across GPUs
config.ft_lr            = 0.05                       # fine-tune learning rate
config.weight_decay     = 5e-4                       # weight decay
config.momentum         = 0.9                        # SGD momentum
config.ft_epochs        = 100                        # total fine-tuning epochs

# —— Evaluation & Logging ——
config.val_targets      = ["fv1_test"]
config.eval_step        = int(config.num_image / config.batch_size + 0.5)

# —— Learning-Rate Schedule ——
config.warmup_epoch     = 5                          # warmup for first 5 epochs

def lr_step_func(epoch):
    """
    Linear warmup (epochs < warmup_epoch), then decay by 0.1 at epochs 30, 60, and 80.
    Returns a multiplier for the base learning rate.
    """
    if epoch < config.warmup_epoch:
        return float(epoch + 1) / config.warmup_epoch
    decay_points = [30, 60, 80]
    return 0.1 ** len([e for e in decay_points if epoch >= e])

config.lr_func = lr_step_func

# —— Teacher Checkpoints ——
config.pretrained_teacher_path        = (
    "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/"
    "output/AdaDistillref/best_backbone_epoch19.pth"
)
# Header path remains unchanged or set later as needed
# config.pretrained_teacher_header_path = "<path_to_header_checkpoint>"

# —— Miscellaneous ——
config.num_workers      = 16                        # DataLoader workers per GPU
