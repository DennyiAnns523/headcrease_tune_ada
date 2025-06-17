from easydict import EasyDict as edict

config = edict()
config.dataset = "emoreIresNet"  # training dataset
config.embedding_size = 512      # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 512         # batch size per GPU
config.num_workers = 16
config.lr = 0.1

# Saving path
config.output = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/output"  # train model output folder

# teacher path
config.pretrained_teacher_path = "output/teacher/295672backbone.pth"
config.pretrained_teacher_header_path = "output/teacher/295672header.pth"  # teacher folder

config.global_step =  34116 #28450
# step to resume

# Margin-penalty loss configurations
config.s = 64.0
config.m = 0.45

# AdaDistill configuration
config.adaptive_alpha = True
config.loss = "ArcFace"        # Option : ArcFace, CosFace, MLLoss

# type of network to train [iresnet100 | iresnet50 | iresnet18 | mobilefacenet]
config.network = "iresnet50"
config.teacher = "iresnet50"
config.SE = False              # SEModule

# Dataset-specific settings
if config.dataset == "emoreIresNet":
    config.rec = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/data/face_recog/faces_emore"
    config.db_file_format = "rec"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 26
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 5686

    def lr_step_func(epoch):
        return ((epoch + 1) / 5) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14, 20, 25] if m - 1 <= epoch]
        )
    config.lr_func = lr_step_func

if config.dataset == "Idifface":
    config.rec = "/media/arjun/New Volume/DL_UbuntuFolderSSD/AdaDLProject/AdaDistill/data/face_recog/faces_emore"
    config.data_path = "./dataset/Idifface"
    config.db_file_format = "folder"
    config.num_classes = 10049
    config.num_image = 502450
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 982 * 4

    def lr_step_func(epoch):
        return ((epoch + 1) / 5) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch]
        )
    config.lr_func = lr_step_func
    config.sample = None  # 50

if config.dataset == "CASIA_WebFace":
    config.rec = "./data/faces_webface_112x112"
    config.db_file_format = "rec"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 60
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 3916

    def lr_step_func(epoch):
        return ((epoch + 1) / 5) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [40, 48, 52] if m - 1 <= epoch]
        )
    config.lr_func = lr_step_func

if config.dataset == "headbandCrease":
    config.data_path = "/home/arjun/Downloads/AdaDLProject/AdaDistill/data/full_augmented_train_fv1"
    config.db_file_format = "folder"
    config.num_classes = 247
    config.num_image = 106210
    config.num_epoch = 100
    config.warmup_epoch = -1
    config.val_targets = ["forehead_verification"]
    config.eval_step = 827
    config.sample = int(1e9)

    def lr_step_func(epoch):
        return ((epoch + 1) / 5) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [10, 20, 25] if m - 1 <= epoch]
        )
    config.lr_func = lr_step_func
