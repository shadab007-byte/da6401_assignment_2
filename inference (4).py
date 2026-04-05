"""Inference and evaluation — DA6401 Assignment 2
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.multitask import MultiTaskPerceptionModel
from losses.iou_loss import IoULoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_dice(logits, targets, eps=1e-6):
    preds = logits.argmax(dim=1)
    num_classes = logits.shape[1]
    dice_list = []
    for c in range(num_classes):
        p = (preds == c).float(); t = (targets == c).float()
        inter = (p * t).sum(); denom = p.sum() + t.sum()
        if denom > 0:
            dice_list.append(((2 * inter + eps) / (denom + eps)).item())
    return float(np.mean(dice_list)) if dice_list else 0.0


def evaluate(args):
    print(f"Device: {DEVICE}")

    model = MultiTaskPerceptionModel(
        classifier_path=args.classifier_path,
        localizer_path=args.localizer_path,
        unet_path=args.unet_path,
    ).to(DEVICE)
    model.eval()

    test_ds = OxfordIIITPetDataset(args.data_root, split="test")
    loader  = DataLoader(test_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=2)

    ce_loss  = torch.nn.CrossEntropyLoss()
    iou_fn   = IoULoss(reduction="mean")

    total_acc, total_iou, total_dice, total_pxacc = 0.0, 0.0, 0.0, 0.0
    n = 0

    with torch.no_grad():
        for imgs, labels, bboxes, masks in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            bboxes = bboxes.to(DEVICE)
            masks  = masks.to(DEVICE)

            out = model(imgs)
            cls_logits = out["classification"]
            loc_preds  = out["localization"]
            seg_logits = out["segmentation"]

            preds = cls_logits.argmax(dim=1)
            acc   = (preds == labels).float().mean().item()
            iou   = 1.0 - iou_fn(loc_preds, bboxes).item()
            dice  = compute_dice(seg_logits, masks)
            pxacc = (seg_logits.argmax(1) == masks).float().mean().item()

            total_acc   += acc
            total_iou   += iou
            total_dice  += dice
            total_pxacc += pxacc
            n += 1

    print(f"\n{'='*50}")
    print(f"Classification Accuracy : {total_acc/n:.4f}")
    print(f"Localization Mean IoU   : {total_iou/n:.4f}")
    print(f"Segmentation Dice Score : {total_dice/n:.4f}")
    print(f"Segmentation Pixel Acc  : {total_pxacc/n:.4f}")
    print(f"{'='*50}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",       type=str, default="data/oxford-iiit-pet")
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--classifier_path", type=str, default="checkpoints/classifier.pth")
    p.add_argument("--localizer_path",  type=str, default="checkpoints/localizer.pth")
    p.add_argument("--unet_path",       type=str, default="checkpoints/unet.pth")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
