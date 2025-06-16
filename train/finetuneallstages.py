'''
STAGE 1=============================================

import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T

from backbones.iresnet import iresnet50
from utils import losses
from config.config_hc import config as cfg
from utils.utils_callbacks_hc import (
    CallBackVerification,
    CallBackLogging,
    CallBackModelCheckpoint
)
from utils.utils_logging import init_logging

torch.backends.cudnn.benchmark = True

# Stage1-only fine-tuning parameters
STAGE1_EPOCHS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 only fine-tuning (head-only)")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank passed from distributed launcher')
    return parser.parse_args()


def main(args):
    # Distributed setup
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Logging / output dir
    if rank == 0:
        os.makedirs(cfg.output, exist_ok=True)
    else:
        time.sleep(2)
    init_logging(logging.getLogger(), rank, cfg.output)
    logging.info(f"Stage-1 fine-tune head-only: epochs={STAGE1_EPOCHS}, rank={rank}")
    if rank == 0:
        print(f">>> STAGE-1 ONLY FINE-TUNE: {STAGE1_EPOCHS} epochs (head frozen) <<<")

    # Data loader with augmentations
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(112, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        T.RandomErasing(p=0.3)
    ])
    from utils.dataset import FaceDatasetFolder
    trainset = FaceDatasetFolder(
        root_dir=cfg.data_path,
        local_rank=args.local_rank,
        number_sample=getattr(cfg, 'sample', None)
    )
    trainset.transform = transform

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=getattr(cfg, 'num_workers', 16),
        pin_memory=True,
        drop_last=True
    )

    # Model & head
    backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).cuda(args.local_rank)
    header   = losses.ArcFace(
        in_features=cfg.embedding_size,
        out_features=cfg.num_classes,
        s=cfg.s, m=cfg.m
    ).cuda(args.local_rank)

    # Load teacher weights
    ckpt = torch.load(cfg.pretrained_teacher_path, map_location='cpu')
    backbone.load_state_dict(ckpt, strict=False)
    logging.info(f"Loaded pretrained teacher from {cfg.pretrained_teacher_path}")

    # Wrap DDP
    model  = DistributedDataParallel(backbone, device_ids=[args.local_rank])
    header = DistributedDataParallel(header,   device_ids=[args.local_rank])

    # Freeze backbone, train head only
    for p in model.parameters(): p.requires_grad = False
    for p in header.parameters(): p.requires_grad = True

    optimizer = torch.optim.SGD(
        [{'params': header.parameters(), 'lr': cfg.ft_lr}],
        momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    total_steps = int(len(trainset)/(cfg.batch_size*world_size)*STAGE1_EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.ft_lr,
        total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    scaler    = GradScaler()
    criterion = CrossEntropyLoss()

    # Callbacks: log per epoch, verify per epoch, checkpoint best
    # compute steps per real epoch
    global_steps_per_epoch = (len(trainset)*world_size)//cfg.batch_size
    callback_log = CallBackLogging(
        frequent=global_steps_per_epoch,
        rank=rank,
        total_step=total_steps,
        batch_size=cfg.batch_size,
        world_size=world_size
    )
    callback_ver = CallBackVerification(
        frequent=global_steps_per_epoch,
        rank=rank,
        val_targets=cfg.val_targets,
        rec_prefix=cfg.verification_path
    )
    callback_ckp = CallBackModelCheckpoint(rank=rank, output=cfg.output)

    global_step = 0

    # Training loop
    for epoch in range(STAGE1_EPOCHS):
        train_sampler.set_epoch(epoch)
        for imgs, labels in loader:
            global_step += 1
            imgs   = imgs.cuda(args.local_rank, non_blocking=True)
            labels = labels.cuda(args.local_rank, non_blocking=True)
            with autocast():
                feats     = model(imgs)
                cos_theta = header(F.normalize(feats, dim=1), labels)
                loss      = criterion(cos_theta, labels)
            scaler.scale(loss).backward()
            clip_grad_norm_(header.parameters(), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # per-epoch log triggers once per epoch
            callback_log(global_step, loss.item(), epoch)

        # end-of-epoch verification & checkpoint
        metrics = callback_ver(global_step, model)
        if metrics:
            callback_ckp(epoch, model, header, metric=metrics[0])

        # explicit epoch completion log
        if rank == 0:
            logging.info(f"=== Completed Epoch {epoch:3d} | global_step={global_step:6d} ===")

    # final verification
    callback_ver(-1, model)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''



'''STAGE 2========================================================
import argparse
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T

from backbones.iresnet import iresnet50
from utils import losses
from config.config_hc import config as cfg
from utils.utils_callbacks_hc import (
    CallBackVerification,
    CallBackLogging,
    CallBackModelCheckpoint
)
from utils.utils_logging import init_logging

torch.backends.cudnn.benchmark = True

# Stage2-only fine-tuning parameters
STAGE2_EPOCHS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 fine-tuning (layer4 + head)")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--backbone_ckpt', type=str, default=cfg.pretrained_teacher_path,
                        help='Path to backbone checkpoint (e.g. best_stage1_backbone)')
    parser.add_argument('--header_ckpt', type=str, default=None,
                        help='(Optional) Path to head checkpoint to resume')
    return parser.parse_args()


def main(args):
    # Distributed setup
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Logging / output dir
    if rank == 0:
        os.makedirs(cfg.output, exist_ok=True)
    else:
        time.sleep(2)
    init_logging(logging.getLogger(), rank, cfg.output)
    logging.info(f"Stage-2 fine-tune layer4+head: epochs={STAGE2_EPOCHS}, rank={rank}")
    if rank == 0:
        print(f">>> STAGE-2 ONLY FINE-TUNE: {STAGE2_EPOCHS} epochs (layer4 + head) <<<")

    # Data loader with augmentations
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(112, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        T.RandomErasing(p=0.3)
    ])
    from utils.dataset import FaceDatasetFolder
    trainset = FaceDatasetFolder(
        root_dir=cfg.data_path,
        local_rank=args.local_rank,
        number_sample=getattr(cfg, 'sample', None)
    )
    trainset.transform = transform

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=getattr(cfg, 'num_workers', 16),
        pin_memory=True,
        drop_last=True
    )

    # Model & head
    backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).cuda(args.local_rank)
    header   = losses.ArcFace(
        in_features=cfg.embedding_size,
        out_features=cfg.num_classes,
        s=cfg.s, m=cfg.m
    ).cuda(args.local_rank)

    # Load backbone checkpoint
    backbone_ckpt = torch.load(args.backbone_ckpt, map_location='cpu')
    backbone.load_state_dict(backbone_ckpt, strict=False)
    logging.info(f"Loaded backbone from {args.backbone_ckpt}")

    # Load header checkpoint if provided
    if args.header_ckpt:
        header_ckpt = torch.load(args.header_ckpt, map_location='cpu')
        header.load_state_dict(header_ckpt)
        logging.info(f"Loaded header from {args.header_ckpt}")

    # Wrap DDP
    model  = DistributedDataParallel(backbone, device_ids=[args.local_rank])
    header = DistributedDataParallel(header,   device_ids=[args.local_rank])

    # Freeze all but layer4 and head
    for name, p in model.named_parameters():
        p.requires_grad = False
    for p in model.module.layer4.parameters():
        p.requires_grad = True
    for p in header.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD([
        {'params': model.module.layer4.parameters(), 'lr': cfg.ft_lr * 0.01},
        {'params': header.parameters(),                     'lr': cfg.ft_lr}
    ], momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    total_steps = int(len(trainset)/(cfg.batch_size*world_size)*STAGE2_EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.ft_lr,
        total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    scaler    = GradScaler()
    criterion = CrossEntropyLoss()

    # Setup callbacks per-epoch
    global_steps_per_epoch = (len(trainset)*world_size)//cfg.batch_size
    callback_log = CallBackLogging(
        frequent=global_steps_per_epoch,
        rank=rank,
        total_step=total_steps,
        batch_size=cfg.batch_size,
        world_size=world_size
    )
    callback_ver = CallBackVerification(
        frequent=global_steps_per_epoch,
        rank=rank,
        val_targets=cfg.val_targets,
        rec_prefix=cfg.verification_path
    )
    callback_ckp = CallBackModelCheckpoint(rank=rank, output=cfg.output)

    global_step = 0

    # Training loop
    for epoch in range(STAGE2_EPOCHS):
        train_sampler.set_epoch(epoch)
        for imgs, labels in loader:
            global_step += 1
            imgs   = imgs.cuda(args.local_rank, non_blocking=True)
            labels = labels.cuda(args.local_rank, non_blocking=True)
            with autocast():
                feats     = model(imgs)
                cos_theta = header(F.normalize(feats, dim=1), labels)
                loss      = criterion(cos_theta, labels)
            scaler.scale(loss).backward()
            clip_grad_norm_(list(model.module.layer4.parameters()) + list(header.parameters()), max_norm=5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # log once per epoch
            callback_log(global_step, loss.item(), epoch)

        # end-of-epoch verification & checkpoint
        metrics = callback_ver(global_step, model)
        if metrics:
            callback_ckp(epoch, model, header, metric=metrics[0])

        if rank == 0:
            logging.info(f"=== Completed Stage2 Epoch {epoch:3d} | global_step={global_step:6d} ===")

    # final verification
    callback_ver(-1, model)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
#STAGE 3====================================================================================
import argparse#
import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as T

from backbones.iresnet import iresnet50
from utils import losses
from config.config_hc import config as cfg
from utils.utils_callbacks_hc import (
    CallBackVerification,
    CallBackLogging,
    CallBackModelCheckpoint
)
from utils.utils_logging import init_logging

torch.backends.cudnn.benchmark = True

# Stage3-only fine-tuning parameters
STAGE3_EPOCHS = 100


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-3 fine-tuning (full backbone + head)")
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank passed from distributed launcher')
    parser.add_argument('--backbone_ckpt', type=str, required=True,
                        help='Path to backbone checkpoint (e.g. best_stage2_backbone)')
    parser.add_argument('--header_ckpt', type=str, default=None,
                        help='(Optional) Path to head checkpoint to resume')
    return parser.parse_args()


def main(args):
    # Distributed setup
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Create output directory on rank 0
    if rank == 0:
        os.makedirs(cfg.output, exist_ok=True)
    else:
        time.sleep(2)
    init_logging(logging.getLogger(), rank, cfg.output)
    logging.info(f"Stage-3 fine-tune full backbone+head: epochs={STAGE3_EPOCHS}, rank={rank}")
    if rank == 0:
        print(f">>> STAGE-3 ONLY FINE-TUNE: {STAGE3_EPOCHS} epochs (full backbone + head) <<<")

    # Data loader with augmentations
    transform = T.Compose([
        T.ToPILImage(),
        T.RandomResizedCrop(112, scale=(0.8,1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomRotation(10),
        T.ToTensor(),
        T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        T.RandomErasing(p=0.3)
    ])
    from utils.dataset import FaceDatasetFolder
    trainset = FaceDatasetFolder(
        root_dir=cfg.data_path,
        local_rank=args.local_rank,
        number_sample=getattr(cfg, 'sample', None)
    )
    trainset.transform = transform

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=getattr(cfg, 'num_workers', 16),
        pin_memory=True,
        drop_last=True
    )

    # Model & head
    backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).cuda(args.local_rank)
    header   = losses.ArcFace(
        in_features=cfg.embedding_size,
        out_features=cfg.num_classes,
        s=cfg.s, m=cfg.m
    ).cuda(args.local_rank)

    # Load backbone checkpoint (Stage2 best)
    backbone_ckpt = torch.load(args.backbone_ckpt, map_location='cpu')
    backbone.load_state_dict(backbone_ckpt, strict=False)
    logging.info(f"Loaded backbone from {args.backbone_ckpt}")

    # Load header checkpoint if provided
    if args.header_ckpt:
        header_ckpt = torch.load(args.header_ckpt, map_location='cpu')
        header.load_state_dict(header_ckpt)
        logging.info(f"Loaded header from {args.header_ckpt}")

    # Wrap DDP
    model  = DistributedDataParallel(backbone, device_ids=[args.local_rank])
    header = DistributedDataParallel(header,   device_ids=[args.local_rank])

    # Unfreeze all backbone and head
    for p in model.parameters():
        p.requires_grad = True
    for p in header.parameters():
        p.requires_grad = True

    # Optimizer with smaller LR on backbone
    optimizer = torch.optim.SGD([
        {'params': model.parameters(),     'lr': cfg.ft_lr * 0.001},
        {'params': header.parameters(),    'lr': cfg.ft_lr}
    ], momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    total_steps = int(len(trainset)/(cfg.batch_size*world_size)*STAGE3_EPOCHS)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.ft_lr,
        total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    scaler    = GradScaler()
    criterion = CrossEntropyLoss()

    # Setup callbacks to run per true epoch
    global_steps_per_epoch = (len(trainset)*world_size)//cfg.batch_size
    callback_log = CallBackLogging(
        frequent=global_steps_per_epoch,
        rank=rank,
        total_step=total_steps,
        batch_size=cfg.batch_size,
        world_size=world_size
    )
    callback_ver = CallBackVerification(
        frequent=global_steps_per_epoch,
        rank=rank,
        val_targets=cfg.val_targets,
        rec_prefix=cfg.verification_path
    )
    callback_ckp = CallBackModelCheckpoint(rank=rank, output=cfg.output)

    global_step = 0

    # Training loop: full backbone + head
    for epoch in range(STAGE3_EPOCHS):
        train_sampler.set_epoch(epoch)
        for imgs, labels in loader:
            global_step += 1
            imgs   = imgs.cuda(args.local_rank, non_blocking=True)
            labels = labels.cuda(args.local_rank, non_blocking=True)
            with autocast():
                feats     = model(imgs)
                cos_theta = header(F.normalize(feats, dim=1), labels)
                loss      = criterion(cos_theta, labels)
            scaler.scale(loss).backward()
            clip_grad_norm_([
                *model.parameters(), *header.parameters()], max_norm=5
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # log end-of-epoch
            callback_log(global_step, loss.item(), epoch)

        # verification & checkpoint at end of epoch
        metrics = callback_ver(global_step, model)
        if metrics:
            callback_ckp(epoch, model, header, metric=metrics[0])

        if rank == 0:
            logging.info(f"=== Completed Stage3 Epoch {epoch:3d} | global_step={global_step:6d} ===")

    # final verification pass
    callback_ver(-1, model)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    main(args)

