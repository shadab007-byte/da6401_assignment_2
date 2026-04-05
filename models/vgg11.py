"""VGG11 encoder
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


# ---------------------------------------------------------------------------
# VGG11 convolutional configuration per the original paper (Simonyan & Zisserman 2014)
# 'M' = MaxPool2d(kernel_size=2, stride=2)
# Numbers = number of output channels for Conv2d(kernel=3, pad=1)
# ---------------------------------------------------------------------------
VGG11_CONFIG = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


def _make_conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 conv -> BN -> ReLU block (no dropout in conv blocks per design choice).

    BatchNorm is placed immediately after Conv2d and before ReLU.
    Justification: BN normalises pre-activation distributions, which
    accelerates convergence and allows higher stable learning rates.
    Dropout is reserved for the fully-connected head where it is most
    effective at preventing co-adaptation of neurons (Srivastava et al. 2014).
    """
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    Architecture follows the VGG11 specification exactly:
      - 8 convolutional layers organised in 5 blocks separated by MaxPool.
      - Channel progression: 64 -> 128 -> 256x2 -> 512x2 -> 512x2
      - Input assumed to be 224x224 (standard VGG input).
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        # ---- Build convolutional blocks ----------------------------------- #
        # We build five named blocks so skip connections can be extracted
        # cleanly for the U-Net decoder.

        # Block 1: 224x224 -> 112x112  (after pool)
        self.block1 = nn.Sequential(
            _make_conv_block(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 112x112 -> 56x56
        self.block2 = nn.Sequential(
            _make_conv_block(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 56x56 -> 28x28
        self.block3 = nn.Sequential(
            _make_conv_block(128, 256),
            _make_conv_block(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 4: 28x28 -> 14x14
        self.block4 = nn.Sequential(
            _make_conv_block(256, 512),
            _make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 5: 14x14 -> 7x7
        self.block5 = nn.Sequential(
            _make_conv_block(512, 512),
            _make_conv_block(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where feature_dict
              contains pre-pool activations for skip connections at each scale.
        """
        # We need pre-pool features for skip connections.
        # Each block is split into conv part and pool part.

        # --- Block 1 ---
        f1 = self._block_conv(self.block1, x)       # [B, 64,  224, 224] before pool
        x = self._block_pool(self.block1, f1)        # [B, 64,  112, 112]

        # --- Block 2 ---
        f2 = self._block_conv(self.block2, x)        # [B, 128, 112, 112]
        x = self._block_pool(self.block2, f2)        # [B, 128,  56,  56]

        # --- Block 3 ---
        f3 = self._block_conv(self.block3, x)        # [B, 256,  56,  56]
        x = self._block_pool(self.block3, f3)        # [B, 256,  28,  28]

        # --- Block 4 ---
        f4 = self._block_conv(self.block4, x)        # [B, 512,  28,  28]
        x = self._block_pool(self.block4, f4)        # [B, 512,  14,  14]

        # --- Block 5 ---
        f5 = self._block_conv(self.block5, x)        # [B, 512,  14,  14]
        bottleneck = self._block_pool(self.block5, f5)  # [B, 512,  7,   7]

        if return_features:
            features = {
                "f1": f1,   # 64 ch,  224x224
                "f2": f2,   # 128 ch, 112x112
                "f3": f3,   # 256 ch,  56x56
                "f4": f4,   # 512 ch,  28x28
                "f5": f5,   # 512 ch,  14x14
            }
            return bottleneck, features

        return bottleneck

    # ---------------------------------------------------------------------- #
    # Helpers to split each sequential block into conv layers and pool layer  #
    # ---------------------------------------------------------------------- #
    @staticmethod
    def _block_conv(block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Run all layers in block EXCEPT the final MaxPool."""
        for layer in block[:-1]:
            x = layer(x)
        return x

    @staticmethod
    def _block_pool(block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Run only the final MaxPool of the block."""
        return block[-1](x)


# Alias so autograder `from models.vgg11 import VGG11` works
VGG11 = VGG11Encoder
