"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class RegressionHead(nn.Module):
    """Bounding box regression head.

    Outputs [x_center, y_center, width, height] in pixel coordinates.
    A Sigmoid activation followed by scaling to image size (224) is used so
    the raw network output is bounded preventing gradient explosion while
    still predicting in pixel space as required.
    """

    def __init__(self, dropout_p: float = 0.5, img_size: int = 224):
        super().__init__()
        self.img_size = img_size
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(1024, 4),
            nn.Sigmoid(),   # outputs in (0, 1); scaled to pixel space below
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        out = self.head(x)                  # (0, 1)
        return out * self.img_size          # scale to pixel space [0, 224]


class VGG11Localizer(nn.Module):
    """VGG11-based localizer"""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = RegressionHead(dropout_p=dropout_p, img_size=224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalized values).
        """
        features = self.encoder(x, return_features=False)   # [B, 512, 7, 7]
        return self.head(features)
