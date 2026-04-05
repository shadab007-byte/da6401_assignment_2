"""Training entrypoint — DA6401 Assignment 2
Trains Task 1 (classifier), Task 2 (localizer), Task 3 (U-Net) sequentially
and logs everything required for the W&B report.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


# ============================================================================ #
# Config
# ============================================================================ #
WANDB_ENTITY  = "iitm_assigment"
WANDB_PROJECT = "da6401-assignment-2"
IMG_SIZE      = 224
NUM_CLASSES   = 37
SEG_CLASSES   = 3   # Oxford trimaps: foreground / background / border
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ #
# Loss helpers
# ============================================================================ #
class DiceLoss(nn.Module):
    """Soft Dice loss for segmentation (multi-class, macro average)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [B, C, H, W]
            targets: [B, H, W] long
        """
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)                    # [B, C, H, W]
        # One-hot encode targets
        one_hot = torch.zeros_like(probs)                       # [B, C, H, W]
        one_hot.scatter_(1, targets.unsqueeze(1), 1)

        dims = (0, 2, 3)   # batch + spatial
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality  = (probs + one_hot).sum(dim=dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class CombinedSegLoss(nn.Module):
    """CE + Dice combined segmentation loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_w   = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits, targets):
        return self.ce_w * self.ce(logits, targets) + self.dice_w * self.dice(logits, targets)


# ============================================================================ #
# Metric helpers
# ============================================================================ #
def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = logits.argmax(dim=1).cpu().numpy()
    tgts  = targets.cpu().numpy()
    f1_per_class = []
    for c in range(num_classes):
        tp = ((preds == c) & (tgts == c)).sum()
        fp = ((preds == c) & (tgts != c)).sum()
        fn = ((preds != c) & (tgts == c)).sum()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        f1_per_class.append(f1)
    return float(np.mean(f1_per_class)) if f1_per_class else 0.0


def compute_iou_metric(pred_boxes: torch.Tensor, tgt_boxes: torch.Tensor, eps: float = 1e-6) -> float:
    """Mean IoU over a batch (NOT the loss, just for logging)."""
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    tgt_x1  = tgt_boxes[:, 0]  - tgt_boxes[:, 2]  / 2
    tgt_y1  = tgt_boxes[:, 1]  - tgt_boxes[:, 3]  / 2
    tgt_x2  = tgt_boxes[:, 0]  + tgt_boxes[:, 2]  / 2
    tgt_y2  = tgt_boxes[:, 1]  + tgt_boxes[:, 3]  / 2
    ix1 = torch.max(pred_x1, tgt_x1); iy1 = torch.max(pred_y1, tgt_y1)
    ix2 = torch.min(pred_x2, tgt_x2); iy2 = torch.min(pred_y2, tgt_y2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    ta = (tgt_x2  - tgt_x1).clamp(0) * (tgt_y2  - tgt_y1).clamp(0)
    union = pa + ta - inter
    return (inter / (union + eps)).mean().item()


def compute_dice_score(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """Macro Dice score (not loss) for logging."""
    num_classes = logits.shape[1]
    preds = logits.argmax(dim=1)   # [B, H, W]
    dice_list = []
    for c in range(num_classes):
        pred_c = (preds == c).float()
        tgt_c  = (targets == c).float()
        inter  = (pred_c * tgt_c).sum()
        denom  = pred_c.sum() + tgt_c.sum()
        if denom == 0:
            continue
        dice_list.append(((2 * inter + eps) / (denom + eps)).item())
    return float(np.mean(dice_list)) if dice_list else 0.0


def compute_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ============================================================================ #
# Data loaders
# ============================================================================ #
def build_loaders(data_root: str, batch_size: int, augment: bool = True):
    """Build train/val/test DataLoaders."""
    aug_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ]) if augment else None

    train_ds = OxfordIIITPetDataset(data_root, split="trainval",
                                    transform=aug_transform, augment=augment)
    test_ds  = OxfordIIITPetDataset(data_root, split="test")

    # Split trainval -> train (90%) + val (10%)
    val_len   = int(0.1 * len(train_ds))
    train_len = len(train_ds) - val_len
    train_ds, val_ds = random_split(train_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ============================================================================ #
# Task 1 — Classification
# ============================================================================ #
def train_classifier(args, train_loader, val_loader):
    """Train VGG11Classifier; logs W&B section 2.1, 2.2, 2.4."""

    # ---- Section 2.2: Three dropout runs ---------------------------------- #
    dropout_configs = [
        ("no_dropout",    0.0),
        ("dropout_p0.2",  0.2),
        ("dropout_p0.5",  0.5),
    ]

    best_model_path = None

    for run_name, dp in dropout_configs:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=f"cls_{run_name}",
            group="Task1_Classification",
            config={
                "task": "classification",
                "dropout_p": dp,
                "epochs": args.cls_epochs,
                "lr": args.cls_lr,
                "batch_size": args.batch_size,
                "run_name": run_name,
            },
            reinit=True,
        )

        model = VGG11Classifier(num_classes=NUM_CLASSES, dropout_p=max(dp, 1e-9)).to(DEVICE)
        # If dp==0 we still build the model; CustomDropout will pass-through when p=0.
        optimizer = optim.Adam(model.parameters(), lr=args.cls_lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(1, args.cls_epochs + 1):
            # -- Train --
            model.train()
            train_loss, train_acc = 0.0, 0.0
            for imgs, labels, _, _ in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc  += compute_accuracy(logits, labels)

            scheduler.step()
            train_loss /= len(train_loader)
            train_acc  /= len(train_loader)

            # -- Val --
            model.eval()
            val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
            all_logits, all_labels = [], []
            with torch.no_grad():
                for imgs, labels, _, _ in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    logits = model(imgs)
                    val_loss += criterion(logits, labels).item()
                    val_acc  += compute_accuracy(logits, labels)
                    all_logits.append(logits.cpu())
                    all_labels.append(labels.cpu())
            val_loss /= len(val_loader)
            val_acc  /= len(val_loader)
            all_logits = torch.cat(all_logits)
            all_labels = torch.cat(all_labels)
            val_f1 = compute_macro_f1(all_logits, all_labels, NUM_CLASSES)

            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/acc":  train_acc,
                "val/loss":   val_loss,
                "val/acc":    val_acc,
                "val/macro_f1": val_f1,
                "lr": scheduler.get_last_lr()[0],
            })

            print(f"[{run_name}] Ep {epoch}/{args.cls_epochs} | "
                  f"TLoss={train_loss:.4f} TAcc={train_acc:.4f} | "
                  f"VLoss={val_loss:.4f} VAcc={val_acc:.4f} F1={val_f1:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if run_name == "dropout_p0.5":   # save the main model
                    best_model_path = "checkpoints/classifier.pth"
                    torch.save({"state_dict": model.state_dict(),
                                "epoch": epoch,
                                "best_metric": best_val_acc}, best_model_path)

        # ---- Section 2.4: Feature maps (only for the p=0.5 run) ---------- #
        if run_name == "dropout_p0.5":
            _log_feature_maps(model, val_loader)

        # ---- Section 2.1: BatchNorm activation distribution -------------- #
        if run_name == "dropout_p0.5":
            _log_bn_activation_dist(args, val_loader)

        wandb.finish()

    return best_model_path


def _log_feature_maps(model, val_loader):
    """Section 2.4 — log first/last conv layer feature maps."""
    model.eval()
    imgs, _, _, _ = next(iter(val_loader))
    img = imgs[:1].to(DEVICE)

    activations = {}

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach().cpu()
        return hook

    # First conv in block1 (index 0 of the Sequential, then index 0 of inner seq)
    h1 = model.encoder.block1[0][0].register_forward_hook(make_hook("first_conv"))
    # Last conv before final pool: block5, second conv block, first Conv2d
    h2 = model.encoder.block5[1][0].register_forward_hook(make_hook("last_conv"))

    with torch.no_grad():
        model(img)
    h1.remove(); h2.remove()

    for name, act in activations.items():
        # act: [1, C, H, W] — log first 16 channels as a grid
        num_channels = min(16, act.shape[1])
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            if i < num_channels:
                fm = act[0, i].numpy()
                ax.imshow(fm, cmap="viridis")
                ax.set_title(f"Ch {i}")
            ax.axis("off")
        plt.suptitle(f"Feature Maps: {name}")
        plt.tight_layout()
        wandb.log({f"feature_maps/{name}": wandb.Image(fig)})
        plt.close(fig)


def _log_bn_activation_dist(args, val_loader):
    """Section 2.1 — compare activation distributions with/without BN."""
    import copy

    imgs, _, _, _ = next(iter(val_loader))
    img = imgs[:16].to(DEVICE)

    results = {}

    for use_bn, label in [(True, "with_BN"), (False, "without_BN")]:
        # Build a tiny model variant just for the 3rd conv layer
        # We approximate 'without BN' by replacing BN with Identity
        model_tmp = VGG11Classifier(num_classes=NUM_CLASSES).to(DEVICE)
        if not use_bn:
            for name, m in model_tmp.named_modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    # Replace BN with identity - we patch parent module
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = dict(model_tmp.named_modules())[parent_name]
                    setattr(parent, child_name, nn.Identity())

        acts = {}
        # Hook on 3rd conv: block2[0][0] is the Conv in block2's first sub-block
        h = model_tmp.encoder.block2[0][0].register_forward_hook(
            lambda mod, inp, out: acts.update({"act": out.detach().cpu()})
        )
        model_tmp.eval()
        with torch.no_grad():
            model_tmp(img)
        h.remove()
        results[label] = acts["act"].flatten().numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, vals in results.items():
        ax.hist(vals, bins=100, alpha=0.6, label=label, density=True)
    ax.set_xlabel("Activation value")
    ax.set_ylabel("Density")
    ax.set_title("3rd Conv Layer Activation Distribution: With vs Without BatchNorm")
    ax.legend()
    wandb.log({"analysis/bn_activation_dist": wandb.Image(fig)})
    plt.close(fig)


# ============================================================================ #
# Task 2 — Localization
# ============================================================================ #
def train_localizer(args, train_loader, val_loader, test_loader):
    """Train VGG11Localizer; logs W&B section 2.5."""

    run = wandb.init(
        project=WANDB_PROJECT, entity=WANDB_ENTITY,
        name="Task2_Localization",
        group="Task2_Localization",
        config={
            "task": "localization",
            "epochs": args.loc_epochs,
            "lr": args.loc_lr,
            "batch_size": args.batch_size,
        },
        reinit=True,
    )

    model = VGG11Localizer(in_channels=3, dropout_p=0.5).to(DEVICE)

    # Optionally load classifier encoder weights for warm-start
    if os.path.exists("checkpoints/classifier.pth"):
        ckpt = torch.load("checkpoints/classifier.pth", map_location=DEVICE)
        state = ckpt.get("state_dict", ckpt)
        enc_state = {k.replace("encoder.", "", 1): v
                     for k, v in state.items() if k.startswith("encoder.")}
        model.encoder.load_state_dict(enc_state, strict=True)
        print("Loaded encoder from classifier checkpoint for localizer warm-start.")

    optimizer = optim.Adam(model.parameters(), lr=args.loc_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    mse_loss  = nn.MSELoss()
    iou_loss  = IoULoss(reduction="mean")

    best_val_iou = 0.0
    best_path = "checkpoints/localizer.pth"

    for epoch in range(1, args.loc_epochs + 1):
        # -- Train --
        model.train()
        t_loss, t_iou = 0.0, 0.0
        for imgs, _, bboxes, _ in train_loader:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = mse_loss(preds, bboxes) + iou_loss(preds, bboxes)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            t_loss += loss.item()
            t_iou  += compute_iou_metric(preds.detach(), bboxes)

        scheduler.step()
        t_loss /= len(train_loader)
        t_iou  /= len(train_loader)

        # -- Val --
        model.eval()
        v_loss, v_iou = 0.0, 0.0
        with torch.no_grad():
            for imgs, _, bboxes, _ in val_loader:
                imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
                preds = model(imgs)
                v_loss += (mse_loss(preds, bboxes) + iou_loss(preds, bboxes)).item()
                v_iou  += compute_iou_metric(preds, bboxes)
        v_loss /= len(val_loader)
        v_iou  /= len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train/loc_loss": t_loss, "train/iou": t_iou,
            "val/loc_loss":   v_loss, "val/iou":   v_iou,
        })
        print(f"[Localizer] Ep {epoch} | TLoss={t_loss:.4f} TIOU={t_iou:.4f} | "
              f"VLoss={v_loss:.4f} VIOU={v_iou:.4f}")

        if v_iou > best_val_iou:
            best_val_iou = v_iou
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch, "best_metric": best_val_iou}, best_path)

    # ---- Section 2.5: Detection table ------------------------------------ #
    _log_detection_table(model, test_loader)

    wandb.finish()
    return best_path


def _log_detection_table(model, test_loader):
    """Section 2.5 — W&B table with bbox predictions overlaid."""
    model.eval()
    iou_fn = IoULoss(reduction="none")

    columns = ["image", "GT bbox (cx,cy,w,h)", "Pred bbox (cx,cy,w,h)", "IoU"]
    table   = wandb.Table(columns=columns)

    collected = 0
    with torch.no_grad():
        for imgs, _, bboxes, _ in test_loader:
            imgs, bboxes = imgs.to(DEVICE), bboxes.to(DEVICE)
            preds = model(imgs)
            ious  = 1.0 - iou_fn(preds, bboxes)   # IoU values

            for i in range(imgs.shape[0]):
                if collected >= 15:
                    break
                # De-normalise image for display
                img_np = _denorm(imgs[i].cpu()).astype(np.uint8)

                # Draw boxes on image
                img_disp = _draw_boxes(img_np,
                                       gt_box=bboxes[i].cpu().numpy(),
                                       pred_box=preds[i].cpu().numpy())
                iou_val = ious[i].item()

                gt_str   = "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*bboxes[i].cpu().tolist())
                pred_str = "[{:.1f},{:.1f},{:.1f},{:.1f}]".format(*preds[i].cpu().tolist())

                table.add_data(wandb.Image(img_disp), gt_str, pred_str, round(iou_val, 4))
                collected += 1
            if collected >= 15:
                break

    wandb.log({"detection/bbox_table": table})


def _denorm(tensor):
    """Convert normalised tensor [3,H,W] to uint8 HWC numpy."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).numpy()
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def _draw_boxes(img_np, gt_box, pred_box):
    """Draw GT (green) and Pred (red) boxes on image; return PIL Image."""
    from PIL import Image as PILImage, ImageDraw
    pil = PILImage.fromarray(img_np)
    draw = ImageDraw.Draw(pil)

    def cx_cy_to_xyxy(box):
        cx, cy, w, h = box
        return cx - w/2, cy - h/2, cx + w/2, cy + h/2

    draw.rectangle(cx_cy_to_xyxy(gt_box),   outline="green", width=2)
    draw.rectangle(cx_cy_to_xyxy(pred_box), outline="red",   width=2)
    return pil


# ============================================================================ #
# Task 3 — Segmentation
# ============================================================================ #
def train_segmentation(args, train_loader, val_loader, test_loader):
    """Train VGG11UNet; logs W&B sections 2.3 and 2.6."""

    # ---- Section 2.3: Three transfer-learning strategies ----------------- #
    strategies = [
        ("strict_frozen",    "freeze_all"),
        ("partial_finetune", "freeze_early"),
        ("full_finetune",    "train_all"),
    ]

    best_model_path = None
    best_dice_overall = 0.0

    for run_name, strategy in strategies:
        run = wandb.init(
            project=WANDB_PROJECT, entity=WANDB_ENTITY,
            name=f"seg_{run_name}",
            group="Task3_Segmentation",
            config={
                "task": "segmentation",
                "strategy": strategy,
                "epochs": args.seg_epochs,
                "lr": args.seg_lr,
                "batch_size": args.batch_size,
            },
            reinit=True,
        )

        model = VGG11UNet(num_classes=SEG_CLASSES).to(DEVICE)

        # Load encoder from classifier checkpoint
        if os.path.exists("checkpoints/classifier.pth"):
            ckpt = torch.load("checkpoints/classifier.pth", map_location=DEVICE)
            state = ckpt.get("state_dict", ckpt)
            enc_state = {k.replace("encoder.", "", 1): v
                         for k, v in state.items() if k.startswith("encoder.")}
            model.encoder.load_state_dict(enc_state, strict=True)

        # Apply freezing strategy
        if strategy == "freeze_all":
            for p in model.encoder.parameters():
                p.requires_grad = False
        elif strategy == "freeze_early":
            # Freeze blocks 1-3, unfreeze 4-5
            for p in model.encoder.block1.parameters(): p.requires_grad = False
            for p in model.encoder.block2.parameters(): p.requires_grad = False
            for p in model.encoder.block3.parameters(): p.requires_grad = False
        # "train_all" — all params trainable (default)

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.seg_lr, weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = CombinedSegLoss()

        best_val_dice = 0.0

        for epoch in range(1, args.seg_epochs + 1):
            # -- Train --
            model.train()
            t_loss, t_dice, t_pxacc = 0.0, 0.0, 0.0
            for imgs, _, _, masks in train_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = criterion(logits, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                t_loss   += loss.item()
                t_dice   += compute_dice_score(logits.detach(), masks)
                t_pxacc  += compute_pixel_accuracy(logits.detach(), masks)

            scheduler.step()
            n = len(train_loader)
            t_loss /= n; t_dice /= n; t_pxacc /= n

            # -- Val --
            model.eval()
            v_loss, v_dice, v_pxacc = 0.0, 0.0, 0.0
            with torch.no_grad():
                for imgs, _, _, masks in val_loader:
                    imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                    logits = model(imgs)
                    v_loss  += criterion(logits, masks).item()
                    v_dice  += compute_dice_score(logits, masks)
                    v_pxacc += compute_pixel_accuracy(logits, masks)
            nv = len(val_loader)
            v_loss /= nv; v_dice /= nv; v_pxacc /= nv

            wandb.log({
                "epoch": epoch,
                "train/seg_loss": t_loss, "train/dice": t_dice, "train/px_acc": t_pxacc,
                "val/seg_loss":   v_loss, "val/dice":   v_dice, "val/px_acc":  v_pxacc,
            })
            print(f"[Seg-{run_name}] Ep {epoch} | TLoss={t_loss:.4f} TDice={t_dice:.4f} | "
                  f"VLoss={v_loss:.4f} VDice={v_dice:.4f} VPxAcc={v_pxacc:.4f}")

            if v_dice > best_val_dice:
                best_val_dice = v_dice
                if strategy == "full_finetune" or v_dice > best_dice_overall:
                    best_dice_overall = v_dice
                    best_model_path = "checkpoints/unet.pth"
                    torch.save({"state_dict": model.state_dict(),
                                "epoch": epoch, "best_metric": best_val_dice},
                               best_model_path)

        # ---- Section 2.6: Segmentation sample images --------------------- #
        if strategy == "full_finetune":
            _log_segmentation_samples(model, val_loader)

        wandb.finish()

    return best_model_path


def _log_segmentation_samples(model, val_loader):
    """Section 2.6 — log 5 images: original / GT mask / predicted mask."""
    model.eval()
    collected = 0

    columns = ["original", "gt_mask", "pred_mask"]
    table   = wandb.Table(columns=columns)

    with torch.no_grad():
        for imgs, _, _, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            preds  = logits.argmax(dim=1)

            for i in range(imgs.shape[0]):
                if collected >= 5:
                    break
                img_np = _denorm(imgs[i].cpu())

                gt_mask   = masks[i].cpu().numpy()
                pred_mask = preds[i].cpu().numpy()

                # Colour-code masks: 0=red, 1=green, 2=blue
                def colorise(m):
                    cmap = np.array([[220, 50, 50], [50, 200, 50], [50, 50, 220]], dtype=np.uint8)
                    return cmap[m]

                table.add_data(
                    wandb.Image(img_np),
                    wandb.Image(colorise(gt_mask)),
                    wandb.Image(colorise(pred_mask)),
                )
                collected += 1
            if collected >= 5:
                break

    wandb.log({"segmentation/sample_table": table})


# ============================================================================ #
# Final pipeline showcase (Section 2.7 is done manually in Colab notebook)
# ============================================================================ #

# ============================================================================ #
# Main
# ============================================================================ #
def parse_args():
    p = argparse.ArgumentParser(description="DA6401 Assignment 2 Training")
    p.add_argument("--data_root",   type=str, default="data/oxford-iiit-pet")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--cls_epochs",  type=int, default=30)
    p.add_argument("--cls_lr",      type=float, default=1e-4)
    p.add_argument("--loc_epochs",  type=int, default=30)
    p.add_argument("--loc_lr",      type=float, default=1e-4)
    p.add_argument("--seg_epochs",  type=int, default=30)
    p.add_argument("--seg_lr",      type=float, default=1e-4)
    p.add_argument("--skip_cls",    action="store_true")
    p.add_argument("--skip_loc",    action="store_true")
    p.add_argument("--skip_seg",    action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("checkpoints", exist_ok=True)
    wandb.login()

    print(f"Using device: {DEVICE}")
    print("Building data loaders...")
    train_loader, val_loader, test_loader = build_loaders(args.data_root, args.batch_size)

    if not args.skip_cls:
        print("\n=== Task 1: Classifier ===")
        train_classifier(args, train_loader, val_loader)

    if not args.skip_loc:
        print("\n=== Task 2: Localizer ===")
        train_localizer(args, train_loader, val_loader, test_loader)

    if not args.skip_seg:
        print("\n=== Task 3: Segmentation ===")
        train_segmentation(args, train_loader, val_loader, test_loader)

    print("\n✅ All tasks complete. Checkpoints saved in ./checkpoints/")


if __name__ == "__main__":
    main()
