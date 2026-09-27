import os.path as osp
import os
import csv
import sys
import time
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel

from config import config
from dataloader.dataloader import get_train_loader, ValPre
from models.builder import EncoderDecoder as segmodel
from dataloader.RGBXDataset import RGBXDataset
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.loss_opr import CEDiceLoss, FocalDiceLoss, ImbalanceAwareWeightedCELoss, FocalLoss2d
from utils.metric import hist_info, compute_score
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, load_model
from utils.transforms import normalize

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

os.environ['MASTER_PORT'] = '169710'

# ──────────────────────────────────────────────
def run_val_epoch(model, device, val_criterion):
    """
    Returns (val_loss, mean_pixel_acc, mean_IoU) computed over the full val split.
    """
    data_setting = {
        'rgb_root':         config.rgb_root_folder,
        'rgb_format':       config.rgb_format,
        'gt_root':          config.gt_root_folder,
        'gt_format':        config.gt_format,
        'transform_gt':     config.gt_transform,
        'x_root':           config.x_root_folder,
        'x_format':         config.x_format,
        'x_single_channel': config.x_is_single_channel,
        'class_names':      config.class_names,
        'train_source':     config.train_source,
        'eval_source':      config.eval_source,
        'test_source':      config.test_source,
    }
    val_dataset = RGBXDataset(data_setting, 'val', ValPre())

    # unwrap DDP ถ้ามี
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    val_loss_sum  = 0.0
    val_loss_cnt  = 0
    hist          = np.zeros((config.num_classes, config.num_classes))
    correct_total = 0
    labeled_total = 0

    H, W = config.image_height, config.image_width  # 512, 512

    with torch.no_grad():
        for i in range(len(val_dataset)):
            sample  = val_dataset[i]
            img     = sample['data']      # numpy HWC uint8
            label   = sample['label']     # numpy HW
            modal_x = sample['modal_x']   # numpy HWC or HW

            # ── resize ให้ตรงกับ input size ──────────────────────────
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
            label_resized = cv2.resize(label.astype(np.uint8), (W, H),
                                       interpolation=cv2.INTER_NEAREST)
            if modal_x.ndim == 2:
                modal_x = cv2.resize(modal_x, (W, H), interpolation=cv2.INTER_LINEAR)
                modal_x = np.stack([modal_x, modal_x, modal_x], axis=2)
            else:
                modal_x = cv2.resize(modal_x, (W, H), interpolation=cv2.INTER_LINEAR)

            # ── normalize → tensor 1CHW ──────────────────────────────
            img_t = torch.from_numpy(
                np.ascontiguousarray(
                    normalize(img, config.norm_mean, config.norm_std)
                    .transpose(2, 0, 1)[np.newaxis]
                )
            ).float().to(device)

            modal_x_t = torch.from_numpy(
                np.ascontiguousarray(
                    normalize(modal_x, config.norm_mean, config.norm_std)
                    .transpose(2, 0, 1)[np.newaxis]
                )
            ).float().to(device)

            gt_t = torch.from_numpy(
                np.ascontiguousarray(label_resized)[np.newaxis]
            ).long().to(device)

            # ── forward (ไม่ส่ง label → คืน logits) ─────────────────
            logits = raw_model(img_t, modal_x_t)       # 1 x C x H x W
            loss_val = val_criterion(logits, gt_t)
            val_loss_sum += loss_val.item()
            val_loss_cnt += 1

            # ── metric ───────────────────────────────────────────────
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # HW
            h, labeled, correct = hist_info(config.num_classes, pred, label_resized)
            hist          += h
            correct_total += correct
            labeled_total += labeled

    raw_model.train()

    avg_val_loss = val_loss_sum / max(val_loss_cnt, 1)
    _, mean_IoU, _, _, mean_pixel_acc, pixel_acc = compute_score(
        hist, correct_total, labeled_total
    )
    return avg_val_loss, mean_pixel_acc, mean_IoU


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()

    cudnn.benchmark = True
    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # data loader
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset)

    is_main = (engine.distributed and engine.local_rank == 0) or (not engine.distributed)

    if is_main:
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        tb = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

        # ── CSV logger ──────────────────────────────────────────────
        os.makedirs(config.log_dir, exist_ok=True)
        csv_path = osp.join(config.log_dir, 'metrics.csv')
        csv_file  = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'epoch', 'lr',
            'train_loss', 'train_pixel_acc',
            'val_loss',   'val_pixel_acc', 'val_mIoU'
        ])
        csv_file.flush()
        logger.info(f'Metrics CSV → {csv_path}')

    class_counts = {
      0: 257138383,       # Background
      1: 64919234,        # Biotite
      2: 161363696,       # Clinopyroxene
      3: 1175084436,      # Hornblende
      4: 414947884,       # K-feldspar
      5: 6728244,         # Leucite
      6: 282291,          # Muscovite
      7: 386404946,       # Olivine
      8: 13222802,        # Opaque minerals
      9: 626824731,       # Orthopyroxene
      10: 618109760,      # Plagioclase
      11: 313176236,      # Quartz
      12: 1518035,        # Spinel
      13: 21730664,       # Topaz
      14: 7323155,        # Tourmaline
    }
   
    #criterion     = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    #criterion     = ImbalanceAwareWeightedCELoss(class_counts=class_counts, num_classes=config.num_classes, ignore_index=config.background)
    #criterion     = FocalDiceLoss(ignore_index = config.background, gamma = 2.0, dice_weight  = 0.3, smooth = 1.0)
    criterion      = CEDiceLoss(ignore_index = config.background, ce_weight = 0.5, dice_weight = 0.5)
    
    val_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)
    if engine.distributed:
        BatchNorm2d = nn.SyncBatchNorm
    else:
        BatchNorm2d = nn.BatchNorm2d

    model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)
    # ----- FHFU models ----- #
    # if config.backbone == 'fhfu':
    #     from models.encoders.DualFHFU import DualFHFU as segmodel
    #     model = segmodel(cfg=config, criterion=criterion,
    #                     encoder=getattr(config, 'fhfu_encoder', 'resnet50'),
    #                     weights=getattr(config, 'fhfu_weights', 'imagenet'))
    # else:
    #     from models.encoders.DualFHFU import EncoderDecoder as segmodel
    #     model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d)    

    # group weight and config optimizer
    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model, BatchNorm2d, base_lr)

    if config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer == 'SGDM':
        optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError

    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        logger.info('.............distributed training.............')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                            output_device=engine.local_rank, find_unused_parameters=False)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    optimizer.zero_grad()
    model.train()
    logger.info('begin training:')

    best_mIoU    = 0.0
    best_epoch   = 0

    for epoch in range(engine.state.epoch, config.nepochs + 1):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        pbar      = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        dataloader = iter(train_loader)

        sum_loss          = 0.0
        train_hist        = np.zeros((config.num_classes, config.num_classes))
        train_correct     = 0
        train_labeled     = 0

        for idx in pbar:
            engine.update_iteration(epoch, idx)

            minibatch = next(dataloader)
            imgs      = minibatch['data'].cuda(non_blocking=True)
            gts       = minibatch['label'].cuda(non_blocking=True)
            modal_xs  = minibatch['modal_x'].cuda(non_blocking=True)

            loss = model(imgs, modal_xs, gts)

            if engine.distributed:
                reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_idx = (epoch - 1) * config.niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr

            # ── accumulate train loss ──
            if engine.distributed:
                iter_loss = reduce_loss.item()
            else:
                iter_loss = loss.item()
            sum_loss += iter_loss

            # ── accumulate train pixel accuracy ──
            with torch.no_grad():
                logits_train = model(imgs, modal_xs)          # 1-pass without criterion
                pred_train   = logits_train.argmax(dim=1).cpu().numpy()   # B x H x W
                gts_np       = gts.cpu().numpy()
                for b in range(pred_train.shape[0]):
                    h, lbl, cor = hist_info(config.num_classes, pred_train[b], gts_np[b])
                    train_hist    += h
                    train_correct += cor
                    train_labeled += lbl

            del loss
            avg_loss = sum_loss / (idx + 1)
            print_str = (f'Epoch {epoch}/{config.nepochs}'
                         f' Iter {idx+1}/{config.niters_per_epoch}:'
                         f' lr={lr:.4e}'
                         f' loss={iter_loss:.4f} avg_loss={avg_loss:.4f}')
            pbar.set_description(print_str, refresh=False)

        if is_main:
            avg_train_loss = sum_loss / len(pbar)

            # train pixel accuracy
            _, _, _, _, train_mean_pixel_acc, _ = compute_score(
                train_hist, train_correct, train_labeled
            )

            # val metrics
            val_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            avg_val_loss, val_mean_pixel_acc, val_mean_IoU = run_val_epoch(
                model, val_device, val_criterion
            )

            logger.info(
                f'[Epoch {epoch}]'
                f'  train_loss={avg_train_loss:.4f}'
                f'  train_acc={train_mean_pixel_acc:.4f}'
                f'  val_loss={avg_val_loss:.4f}'
                f'  val_acc={val_mean_pixel_acc:.4f}'
                f'  val_mIoU={val_mean_IoU:.4f}'
            )

            # ── Best Model ───────────────────────────────────────────
            if val_mean_IoU > best_mIoU:
                best_mIoU  = val_mean_IoU
                best_epoch = epoch
                best_path  = osp.join(config.checkpoint_dir, 'epoch-best.pth')
                engine.save_checkpoint(best_path)
                logger.info(f'  ★ New best mIoU={best_mIoU:.4f} → saved to {best_path}')

            # ── TensorBoard ──
            tb.add_scalar('Loss/train',         avg_train_loss,       epoch)
            tb.add_scalar('Loss/val',            avg_val_loss,         epoch)
            tb.add_scalar('Accuracy/train',      train_mean_pixel_acc, epoch)
            tb.add_scalar('Accuracy/val',        val_mean_pixel_acc,   epoch)
            tb.add_scalar('mIoU/val',            val_mean_IoU,         epoch)
            tb.add_scalar('LR',                  lr,                   epoch)

            # ── CSV ──
            csv_writer.writerow([
                epoch, f'{lr:.6e}',
                f'{avg_train_loss:.6f}',  f'{train_mean_pixel_acc:.6f}',
                f'{avg_val_loss:.6f}',    f'{val_mean_pixel_acc:.6f}',
                f'{val_mean_IoU:.6f}',
            ])
            csv_file.flush()

        # ── checkpoint ──
        if (epoch >= config.checkpoint_start_epoch and epoch % config.checkpoint_step == 0) \
                or epoch == config.nepochs:
            if engine.distributed and engine.local_rank == 0:
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.checkpoint_dir, config.log_dir, config.log_dir_link)

    if is_main:
        csv_file.close()
        logger.info('Training finished. Metrics saved to ' + csv_path)
        logger.info(f'Best model → Epoch {best_epoch}  mIoU={best_mIoU:.4f}  '
                    f'saved at {osp.join(config.checkpoint_dir, "epoch-best.pth")}')