# utils_callbacks_hc.py

import os
import logging
import time
from typing import List

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

#import eval.verification_hc
#from verification_hc  import load_bin, test
# new
from eval.verification_hc import load_bin, test

from backbones.iresnet import iresnet50
from config.config_hc import config as cfg


class NormalizeBackbone(torch.nn.Module):
    """
    Wraps a backbone model to return unit-normalized embeddings.
    """
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return F.normalize(feats, dim=1)


class CallBackVerification(object):
    """
    Evaluation callback: runs verification on .bin protocols at a specified frequency.
    """
    def __init__(
        self,
        frequent: int,
        rank: int,
        val_targets: List[str],
        rec_prefix: str,
        image_size=(112, 112)
    ):
        self.frequent = frequent
        self.rank = rank
        self.highest_acc_list = [0.0] * len(val_targets)
        self.ver_list = []
        self.ver_name_list = []
        if self.rank == 0:
            self.init_dataset(val_targets, rec_prefix, image_size)

    def init_dataset(
        self,
        val_targets: List[str],
        rec_prefix: str,
        image_size
    ):
        for name in val_targets:
            path = os.path.join(rec_prefix, f"{name}.bin")
            if os.path.exists(path):
                data_set = load_bin(path, image_size)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)
            else:
                logging.warning(f"Verification protocol not found: {path}")

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        backbone.train()
        norm_model = NormalizeBackbone(backbone).eval()

        with torch.no_grad(), autocast():
            for idx, data_set in enumerate(self.ver_list):
                name = self.ver_name_list[idx]
                # use batch size from config
                _, _, acc2, std2, xnorm, _ = test(
                    data_set,
                    norm_model,
                    batch_size=cfg.batch_size,
                    nfolds=10
                )
                logging.info(f"[{name}][{global_step}] Accuracy: {acc2:.5f} ± {std2:.5f}")
                if acc2 > self.highest_acc_list[idx]:
                    self.highest_acc_list[idx] = acc2
                    logging.info(
                        f"[{name}][{global_step}] New best: {acc2:.5f}"
                    )
                results.append(acc2)

        backbone.train()
        return results

    def __call__(self, num_update: int, backbone: torch.nn.Module):
        if self.rank != 0:
            return
        if (num_update > 0 and num_update % self.frequent == 0) or num_update == -1:
            return self.ver_test(backbone, num_update)


class CallBackLogging(object):
    """
    Logging callback: prints loss, epoch, step at a specified frequency.
    """
    def __init__(
        self,
        frequent: int,
        rank: int,
        total_step: int,
        batch_size: int,
        world_size: int,
        writer=None
    ):
        self.frequent = frequent
        self.rank = rank
        self.time_start = time.time()
        self.total_step = total_step
        self.batch_size = batch_size
        self.world_size = world_size
        self.writer = writer
        self.initialized = False
        self.tic = None

    def __call__(
        self,
        global_step: int,
        loss_meter,
        epoch: int
    ):
        if self.rank != 0 or global_step <= 0 or global_step % self.frequent != 0:
            return

        if not self.initialized:
            self.initialized = True
            self.tic = time.time()
            return

        elapsed = time.time() - self.tic
        speed = (
            self.frequent * self.batch_size * self.world_size / elapsed
        ) if elapsed > 0 else float('inf')
        time_now = (time.time() - self.time_start) / 3600
        time_total = time_now / (global_step / self.total_step)
        time_for_end = time_total - time_now

        # extract numeric loss
        if hasattr(loss_meter, 'avg'):
            loss_value = float(loss_meter.avg)
        elif hasattr(loss_meter, 'val'):
            loss_value = float(loss_meter.val)
        elif isinstance(loss_meter, (float, int)):
            loss_value = float(loss_meter)
        else:
            logging.warning(
                "CallBackLogging: could not extract numeric from loss_meter"
            )
            loss_value = 0.0

        if self.writer:
            self.writer.add_scalar('time_for_end', time_for_end, global_step)
            self.writer.add_scalar('loss', loss_value, global_step)

        logging.info(
            f"[Epoch {epoch:3d} | Step {global_step:6d}]  "
            f"Speed {speed:.2f} samples/sec  Loss {loss_value:.4f}  ETA {time_for_end:.2f}h"
        )
        self.tic = time.time()


class CallBackModelCheckpoint(object):
    """
    Checkpoint callback: saves model & header when the metric improves.
    """
    def __init__(
        self,
        rank: int,
        output: str = cfg.output
    ):
        self.rank = rank
        self.output = output
        self.best_metric = -float('inf')

    def __call__(
        self,
        epoch: int,
        backbone: torch.nn.Module,
        header: torch.nn.Module = None,
        metric: float = None
    ):
        if self.rank != 0 or metric is None:
            return

        if metric > self.best_metric:
            old_best = self.best_metric
            self.best_metric = metric

            backbone_path = os.path.join(
                self.output, f"best_backbone_epoch{epoch:02d}.pth"
            )
            torch.save(
                backbone.module.state_dict(),
                backbone_path
            )
            logging.info(
                f"[Metric {old_best:.5f} → {metric:.5f}] "
                f"Saved new best backbone: {backbone_path}"
            )

            if header is not None:
                header_path = os.path.join(
                    self.output, f"best_header_epoch{epoch:02d}.pth"
                )
                torch.save(
                    header.module.state_dict(),
                    header_path
                )
                logging.info(
                    f"[Metric {old_best:.5f} → {metric:.5f}] "
                    f"Saved new best header:   {header_path}"
                )
