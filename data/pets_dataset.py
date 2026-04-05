"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# Oxford-IIIT Pet trimap pixel values:
#   1 = foreground (pet)
#   2 = background
#   3 = not classified / border
# We remap to 0-indexed class labels for CrossEntropyLoss
TRIMAP_REMAP = {1: 0, 2: 1, 3: 2}


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Returns per sample:
        image       : FloatTensor [3, 224, 224]  (normalised)
        label       : int  (breed class 0-36)
        bbox        : FloatTensor [4]  (cx, cy, w, h) in pixel space
        mask        : LongTensor [224, 224] with values in {0, 1, 2}
    """

    IMG_SIZE = 224   # fixed VGG11 input size

    # ImageNet normalisation (used because VGG11 was originally trained on ImageNet)
    MEAN = (0.485, 0.456, 0.406)
    STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        transform=None,
        augment: bool = False,
    ):
        """
        Args:
            root:      Root directory that contains 'images/', 'annotations/' sub-dirs.
            split:     'trainval' or 'test'.
            transform: Optional additional albumentations transform (applied after
                       the mandatory resize + normalise).
            augment:   If True apply random horizontal flip + colour jitter.
        """
        super().__init__()
        self.root    = Path(root)
        self.split   = split
        self.augment = augment
        self.transform = transform

        # Parse split file
        split_file = self.root / "annotations" / f"{split}.txt"
        self.samples = []   # list of (img_stem, class_id_1indexed, species, breed_id)
        with open(split_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                stem        = parts[0]          # e.g. "Abyssinian_1"
                class_id    = int(parts[1]) - 1  # 1-indexed -> 0-indexed (0..36)
                self.samples.append((stem, class_id))

        # Build annotation paths
        self.ann_dir  = self.root / "annotations"
        self.img_dir  = self.root / "images"
        self.xml_dir  = self.ann_dir / "xmls"
        self.mask_dir = self.ann_dir / "trimaps"

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _load_image(self, stem: str) -> Image.Image:
        path = self.img_dir / f"{stem}.jpg"
        img = Image.open(path).convert("RGB")
        return img

    def _load_mask(self, stem: str, target_size: Tuple[int, int]) -> np.ndarray:
        path = self.mask_dir / f"{stem}.png"
        mask = Image.open(path)
        # Nearest-neighbour resize to preserve label values
        mask = mask.resize(target_size, Image.NEAREST)
        mask_np = np.array(mask, dtype=np.int64)
        # Remap trimap values to 0-indexed class labels
        out = np.zeros_like(mask_np)
        for src, dst in TRIMAP_REMAP.items():
            out[mask_np == src] = dst
        return out

    def _load_bbox(self, stem: str, orig_w: int, orig_h: int) -> Optional[np.ndarray]:
        """Load bbox from XML; returns [cx, cy, w, h] scaled to IMG_SIZE pixels."""
        xml_path = self.xml_dir / f"{stem}.xml"
        if not xml_path.exists():
            # Return a dummy full-image box if annotation is missing
            return np.array([self.IMG_SIZE / 2, self.IMG_SIZE / 2,
                              self.IMG_SIZE, self.IMG_SIZE], dtype=np.float32)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj  = root.find("object")
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        # Scale to IMG_SIZE
        sx = self.IMG_SIZE / orig_w
        sy = self.IMG_SIZE / orig_h
        xmin, xmax = xmin * sx, xmax * sx
        ymin, ymax = ymin * sy, ymax * sy

        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w  = xmax - xmin
        h  = ymax - ymin
        return np.array([cx, cy, w, h], dtype=np.float32)

    def _normalise(self, img_np: np.ndarray) -> torch.Tensor:
        """HWC uint8 -> CHW float normalised tensor."""
        img = img_np.astype(np.float32) / 255.0
        mean = np.array(self.MEAN, dtype=np.float32)
        std  = np.array(self.STD,  dtype=np.float32)
        img  = (img - mean) / std
        return torch.from_numpy(img.transpose(2, 0, 1))   # [3, H, W]

    # ------------------------------------------------------------------ #
    # Dataset interface
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        stem, class_id = self.samples[idx]

        # Load image
        img_pil = self._load_image(stem)
        orig_w, orig_h = img_pil.size

        # Resize image
        img_pil = img_pil.resize((self.IMG_SIZE, self.IMG_SIZE), Image.BILINEAR)
        img_np  = np.array(img_pil, dtype=np.uint8)

        # Optionally augment with albumentations
        mask_np = self._load_mask(stem, (self.IMG_SIZE, self.IMG_SIZE))

        if self.augment and self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np    = augmented["image"]
            mask_np   = augmented["mask"]
        elif self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_np    = augmented["image"]
            mask_np   = augmented["mask"]

        # Normalise
        img_tensor  = self._normalise(img_np)

        # Bounding box
        bbox = self._load_bbox(stem, orig_w, orig_h)
        bbox_tensor = torch.from_numpy(bbox)

        # Mask
        mask_tensor = torch.from_numpy(mask_np.astype(np.int64))

        return img_tensor, class_id, bbox_tensor, mask_tensor
