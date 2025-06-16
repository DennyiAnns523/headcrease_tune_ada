import argparse
import logging
import os
import time
import glob
import re

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import GradScaler, autocast

from backbones.mobilefacenet import MobileFaceNet
from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, iresnet18

torch.backends.cudnn.benchmark = True

def main(args):
    # --- Distributed setup ---
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {local_rank}] Finished NCCL init")

    # --- Output directory & logging init ---
    if not os.path.exists(cfg.output) and rank == 0:
        os.makedirs(cfg.output)
    else:
        time.sleep(2)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)

    # --- Dataset loading ---
    if cfg.db_file_format != "rec":
        trainset = FaceDatasetFolder(
            root_dir=cfg.data_path,
            local_rank=local_rank,
            number_sample=cfg.sample
        )
    else:
        trainset = MXFaceDataset(
            root_dir=cfg.rec,
            local_rank=local_rank
        )
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)
    logging.info("Total dataset length: %d", len(trainset))
    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        pin_memory=False,
        drop_last=True
    )

    # --- Model & resume load ---
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "mobilefacenet":
        backbone = MobileFaceNet(input_size=(112, 112)).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet18":
        backbone = iresnet18(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        logging.info("load backbone failed!")
        exit()

    # if resuming, load the latest best backbone checkpoint
    if args.resume:
        pattern = os.path.join(cfg.output, "best_backbone_epoch*.pth")
        all_ckpts = sorted(glob.glob(pattern))
        if all_ckpts:
            latest = all_ckpts[-1]
            m = re.search(r"epoch(\d+)", os.path.basename(latest))
            resume_epoch = int(m.group(1)) if m else 0
            backbone.load_state_dict(
                torch.load(latest, map_location=torch.device(local_rank))
            )
            if rank == 0:
                logging.info(f"Resumed backbone from {latest} (epoch {resume_epoch})")
        else:
            logging.info("No best_backbone checkpoint found; starting from scratch")

    # --- Broadcast & wrap DDP ---
    if world_size > 1:
        for ps in backbone.parameters():
            dist.broadcast(ps.data, 0)
    backbone = DistributedDataParallel(backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # --- Header & resume load ---
    if cfg.loss == "ArcFace":
        header = losses.ArcFace(
            in_features=cfg.embedding_size,
            out_features=cfg.num_classes,
            s=cfg.s,
            m=cfg.m
        ).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.CosFace(
            in_features=cfg.embedding_size,
            out_features=cfg.num_classes,
            s=cfg.s,
            m=cfg.m
        ).to(local_rank)
    else:
        print("Header not implemented")
        exit()

    # if resuming, load the latest best header checkpoint
    if args.resume:
        pattern_h = os.path.join(cfg.output, "best_header_epoch*.pth")
        all_h = sorted(glob.glob(pattern_h))
        if all_h:
            latest_h = all_h[-1]
            header.load_state_dict(
                torch.load(latest_h, map_location=torch.device(local_rank))
            )
            if rank == 0:
                logging.info(f"Resumed header from {latest_h}")
        else:
            logging.info("No best_header checkpoint found; starting from scratch")

    torch.cuda.empty_cache()
    header = DistributedDataParallel(header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    # --- Optimizers & schedulers ---
    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9,
        weight_decay=cfg.weight_decay
    )
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(opt_header, lr_lambda=cfg.lr_func)
    criterion = CrossEntropyLoss()

    # --- Steps & epochs bookkeeping ---
    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0:
        logging.info("Total Step is: %d", total_step)

    # if you want to resume schedule by step count, you can recover start_epoch from cfg.global_step here
    if args.resume and 'resume_epoch' in locals():
        start_epoch = resume_epoch
        scheduler_backbone.last_epoch = start_epoch
        scheduler_header.last_epoch = start_epoch
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]
        logging.info(f"Resuming training from epoch {start_epoch}")

    # --- Callbacks ---
    if cfg.db_file_format == "rec":
        verif_data_dir = cfg.rec
    else:
        verif_data_dir = cfg.data_path

    callback_verification = CallBackVerification(
        cfg.eval_step,
        rank,
        cfg.val_targets,
        rec_prefix=verif_data_dir,
        image_size=(112, 112)
    )
    callback_logging    = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)

    loss_meter = AverageMeter()
    global_step = cfg.global_step
    scaler = GradScaler()

    # --- Training loop ---
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for batch_idx, (img, label) in enumerate(train_loader):
            global_step += 1
            img   = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            with autocast():
                features   = F.normalize(backbone(img))
                thetas     = header(features, label)
                loss_value = criterion(thetas, label)

            scaler.scale(loss_value).backward()
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            scaler.step(opt_backbone)
            scaler.step(opt_header)
            scaler.update()
            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss_meter.update(loss_value.item(), 1)

            # Logging
            callback_logging(global_step, loss_meter, epoch)

            # Verification + conditional checkpoint
            accuracies = callback_verification(global_step, backbone)
            if accuracies:
                best_acc = accuracies[0]
                callback_checkpoint(epoch, backbone, header, metric=best_acc)

        scheduler_backbone.step()
        scheduler_header.step()

    # final full verification
    callback_verification(-1, backbone)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume',     type=int, default=0, help="resume training")
    args = parser.parse_args()
    torch.cuda.set_device(args.local_rank)
    main(args)
