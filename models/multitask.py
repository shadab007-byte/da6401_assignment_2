"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .classification import ClassificationHead
from .localization import RegressionHead
from .segmentation import VGG11UNet


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    A single VGG11 encoder feeds three task heads:
      1. ClassificationHead  -> 37-class breed logits
      2. RegressionHead      -> 4-value bounding box [cx, cy, w, h] in pixels
      3. U-Net decoder       -> pixel-wise segmentation logits [B, 3, H, W]
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
    ):
        import gdown
        gdown.download(id="<classifier.pth drive id>", output=classifier_path, quiet=False)
        gdown.download(id="<localizer.pth drive id>", output=localizer_path, quiet=False)
        gdown.download(id="<unet.pth drive id>", output=unet_path, quiet=False)

        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder  = VGG11Encoder(in_channels=in_channels)
        self.cls_head = ClassificationHead(num_classes=num_breeds)
        self.loc_head = RegressionHead(img_size=224)
        self._unet    = VGG11UNet(num_classes=seg_classes, in_channels=in_channels)

        self._load_checkpoint(classifier_path, device, task="classifier")
        self._load_checkpoint(localizer_path,  device, task="localizer")
        self._load_checkpoint(unet_path,       device, task="unet")

        # Share the encoder across all tasks
        self._unet.encoder = self.encoder

    def _load_checkpoint(self, path: str, device, task: str):
        if not os.path.exists(path):
            print(f"[MultiTask] Warning: '{path}' not found, skipping.")
            return
        ckpt  = torch.load(path, map_location=device)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        if task == "classifier":
            enc_state  = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
            head_state = {k[len("head."):]:    v for k, v in state.items() if k.startswith("head.")}
            self.encoder.load_state_dict(enc_state, strict=False)
            self.cls_head.load_state_dict(head_state, strict=False)
        elif task == "localizer":
            head_state = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
            self.loc_head.load_state_dict(head_state, strict=False)
        elif task == "unet":
            self._unet.load_state_dict(state, strict=False)

        print(f"[MultiTask] Loaded {task} from '{path}'")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor
        """
        bottleneck, features = self.encoder(x, return_features=True)

        cls_logits = self.cls_head(bottleneck)
        loc_out    = self.loc_head(bottleneck)

        z = self._unet.bottleneck_drop(bottleneck)
        z = self._unet.up1(z);   z = torch.cat([z, features["f5"]], dim=1); z = self._unet.conv1(z)
        z = self._unet.up2(z);   z = torch.cat([z, features["f4"]], dim=1); z = self._unet.conv2(z)
        z = self._unet.up3(z);   z = torch.cat([z, features["f3"]], dim=1); z = self._unet.conv3(z)
        z = self._unet.up4(z);   z = torch.cat([z, features["f2"]], dim=1); z = self._unet.conv4(z)
        z = self._unet.up5(z);   z = torch.cat([z, features["f1"]], dim=1); z = self._unet.conv5(z)
        seg_logits = self._unet.final_conv(z)

        return {
            "classification": cls_logits,
            "localization":   loc_out,
            "segmentation":   seg_logits,
        }
