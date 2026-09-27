import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualizer import print_iou, show_img 
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBXDataset
from models.builder import EncoderDecoder as segmodel
from dataloader.dataloader import ValPre

logger = get_logger()

def save_confusion_matrix(hist, class_names, save_dir):
    """Save full 15×15 confusion matrix + Leucite-focused plots"""
    os.makedirs(save_dir, exist_ok=True)
    n = len(class_names)

    # Save raw numpy
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), hist)

    # Row-normalize
    row_sum   = hist.sum(axis=1, keepdims=True)
    row_sum   = np.where(row_sum == 0, 1, row_sum)
    hist_norm = hist.astype(float) / row_sum

    leucite_idx  = class_names.index('Leucite') if 'Leucite' in class_names else -1
    minority     = ['Leucite', 'Muscovite', 'Spinel']
    minority_idx = [class_names.index(c) for c in minority if c in class_names]

    # Custom annotation: "norm\n(raw)"
    annot = np.empty((n, n), dtype=object)
    for r in range(n):
        for c in range(n):
            raw = int(hist[r, c])
            pct = hist_norm[r, c]
            annot[r, c] = '0' if raw == 0 else f'{pct:.2f}\n({raw:,})'

    # Top off-diagonal misclassify (>1%) per row
    top_off = {}
    for r in range(n):
        row = hist_norm[r].copy()
        row[r] = 0
        for c in np.argsort(row)[::-1][:3]:
            if hist_norm[r, c] > 0.01:
                top_off[(r, c)] = hist_norm[r, c]

    # ── Plot 1: Full normalized 15×15 ────────────────────────────
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(hist_norm,
                annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax,
                cbar_kws={'shrink': 0.6, 'label': 'Proportion (row-normalized)'},
                annot_kws={'size': 7},
                linewidths=0.4, linecolor='#e2e8f0',
                vmin=0, vmax=1)

    for i in range(n):
        ax.add_patch(plt.Rectangle((i, i), 1, 1,
                                    fill=True, facecolor='#DCFCE7',
                                    edgecolor='#16A34A', lw=1.5, zorder=2))
    if leucite_idx >= 0:
        ax.add_patch(plt.Rectangle((0, leucite_idx), n, 1,
                                    fill=False, edgecolor='#DC2626', lw=3, zorder=4))
        ax.add_patch(plt.Rectangle((leucite_idx, 0), 1, n,
                                    fill=False, edgecolor='#DC2626', lw=3, zorder=4))
    for mi in minority_idx:
        if mi != leucite_idx:
            ax.add_patch(plt.Rectangle((0, mi), n, 1,
                                        fill=False, edgecolor='#D97706', lw=2, zorder=3))
    for (r, c) in top_off:
        ax.add_patch(plt.Rectangle((c, r), 1, 1,
                                    fill=False, edgecolor='#7C3AED',
                                    lw=1.8, linestyle='--', zorder=3))

    ax.set_xlabel('Predicted class', fontsize=13, labelpad=8)
    ax.set_ylabel('Ground truth class', fontsize=13, labelpad=8)
    ax.set_title(
        'Confusion matrix — 15×15 per-pixel (row-normalized)\n'
        'Annotation: proportion (raw pixel count)',
        fontsize=13, fontweight='bold', pad=14)
    ax.tick_params(axis='x', rotation=35, labelsize=9)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)

    legend_patches = [
        mpatches.Patch(fc='#DCFCE7', ec='#16A34A', lw=1.5, label='Diagonal — True Positive'),
        mpatches.Patch(fc='none',    ec='#DC2626', lw=2.5, label='Leucite row/col (IoU=0%)'),
        mpatches.Patch(fc='none',    ec='#D97706', lw=2,   label='Minority class row'),
        mpatches.Patch(fc='none',    ec='#7C3AED', lw=1.8, ls='--',
                       label='Top misclassification (>1%)'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              bbox_to_anchor=(1.3, 1.0), fontsize=18, framealpha=0.9)

    plt.tight_layout()
    out = os.path.join(save_dir, 'confusion_matrix_normalized.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f'[confusion] Saved: {out}')

    # ── Plot 2: Raw pixel count ───────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(22, 18))
    sns.heatmap(hist.astype(float),
                annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2,
                cbar_kws={'shrink': 0.6, 'label': 'Pixel count (raw)'},
                annot_kws={'size': 7},
                linewidths=0.4, linecolor='#e2e8f0')
    for i in range(n):
        ax2.add_patch(plt.Rectangle((i, i), 1, 1,
                                     fill=False, edgecolor='#16A34A', lw=2, zorder=3))
    if leucite_idx >= 0:
        ax2.add_patch(plt.Rectangle((0, leucite_idx), n, 1,
                                     fill=False, edgecolor='#DC2626', lw=3, zorder=4))
        ax2.add_patch(plt.Rectangle((leucite_idx, 0), 1, n,
                                     fill=False, edgecolor='#DC2626', lw=3, zorder=4))
    ax2.set_xlabel('Predicted class', fontsize=13, labelpad=8)
    ax2.set_ylabel('Ground truth class', fontsize=13, labelpad=8)
    ax2.set_title('Confusion matrix — raw pixel counts\nGreen diagonal = TP · Red = Leucite',
                  fontsize=13, fontweight='bold', pad=14)
    ax2.tick_params(axis='x', rotation=35, labelsize=9)
    ax2.tick_params(axis='y', rotation=0,  labelsize=9)
    plt.tight_layout()
    out2 = os.path.join(save_dir, 'confusion_matrix_raw.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f'[confusion] Saved: {out2}')

    # ── Plot 3: Leucite misclassification bar ─────────────────────
    if leucite_idx >= 0:
        counts = hist[leucite_idx].copy()
        total  = counts.sum()
        pct    = counts / max(total, 1) * 100
        sort_i = np.argsort(counts)[::-1]

        colors = ['#16A34A' if i == leucite_idx else '#DC2626' for i in sort_i]
        fig3, ax3 = plt.subplots(figsize=(14, 5))
        ax3.bar(range(n), pct[sort_i], color=colors, edgecolor='white')
        ax3.set_xticks(range(n))
        ax3.set_xticklabels([class_names[i] for i in sort_i],
                             rotation=30, ha='right', fontsize=10)
        ax3.set_ylabel('% of Leucite GT pixels', fontsize=11)
        ax3.set_title(
            f'Leucite misclassification breakdown\n'
            f'Total Leucite GT pixels: {int(total):,}  |  '
            f'TP: {int(counts[leucite_idx]):,} ({pct[leucite_idx]:.1f}%)',
            fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        tp_p = mpatches.Patch(color='#16A34A', label='Correct (TP)')
        fp_p = mpatches.Patch(color='#DC2626', label='Misclassified')
        ax3.legend(handles=[tp_p, fp_p], fontsize=10)
        plt.tight_layout()
        out3 = os.path.join(save_dir, 'leucite_misclassification.png')
        plt.savefig(out3, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f'[confusion] Saved: {out3}')

    logger.info(f'[confusion] All plots saved to: {save_dir}/')


# ── SegEvaluator ─────────────────────────────────────────────────────

class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img     = data['data']
        label   = data['label']
        modal_x = data['modal_x']
        name    = data['fn']

        pred = self.sliding_eval_rgbX(
            img, modal_x, config.eval_crop_size, config.eval_stride_rate, device)

        # ensure pred and label have same shape before hist_info
        if pred.shape != label.shape:
            pred = cv2.resize(pred.astype(np.uint8),
                              (label.shape[1], label.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(pred.dtype)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path + '_color')
            fn = name + '.png'

            result_img   = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path + '_color', fn))

            cv2.imwrite(os.path.join(self.save_path, fn), pred)
            logger.info('Save the image ' + fn)

        if self.show_image:
            colors   = self.dataset.get_class_colors()
            image    = img
            clean    = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean, label, pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist    = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count   = 0
        for d in results:
            hist    += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count   += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = \
            compute_score(hist, correct, labeled)

        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                dataset.class_names, show_no_back=False)

        # expose numeric values
        self.last_metrics = {
            'mean_IoU':       float(mean_IoU),
            'mean_pixel_acc': float(mean_pixel_acc),
            'pixel_acc':      float(pixel_acc),
            'freq_IoU':       float(freq_IoU),
        }

        # Save confusion matrix automatically
        cm_dir = os.path.join(
            os.path.dirname(config.checkpoint_dir), 'confusion_matrix')
        save_confusion_matrix(hist, dataset.class_names, cm_dir)

        return result_line

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs',     default='last', type=str)
    parser.add_argument('-d', '--devices',    default='0',    type=str)
    parser.add_argument('-v', '--verbose',    default=False,  action='store_true')
    parser.add_argument('--show_image', '-s', default=False,  action='store_true')
    parser.add_argument('--save_path',  '-p', default=None)
    parser.add_argument('--split',            default='test',
                        choices=['val', 'test'],
                        help='Dataset split to evaluate on')

    args    = parser.parse_args()
    all_dev = parse_devices(args.devices)

    network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    # ---- Use FHFU ---- #
    # if config.backbone == 'fhfu':
    #     from models.encoders.DualFHFU import DualFHFU as segmodel
    #     network = segmodel(cfg=config, criterion=None,
    #                       encoder=getattr(config, 'fhfu_encoder', 'resnet50'),
    #                       weights=getattr(config, 'fhfu_weights', 'imagenet'))
    # else:
    #     from models.encoders.builder import EncoderDecoder as segmodel
    #     network = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
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
    val_pre = ValPre()
    dataset = RGBXDataset(data_setting, args.split, val_pre)
    logger.info(f'Evaluating on split: {args.split} ({len(dataset)} samples)')

    with torch.no_grad():
        segmentor = SegEvaluator(
            dataset, config.num_classes, config.norm_mean,
            config.norm_std, network,
            config.eval_scale_array, config.eval_flip,
            all_dev, args.verbose, args.save_path,
            args.show_image)
        segmentor.run(config.checkpoint_dir, args.epochs,
                      config.val_log_file, config.link_val_log_file)