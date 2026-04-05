"""Classification components
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class ClassificationHead(nn.Module):
    """Fully-connected classification head for VGG11.

    Design justification for BN + Dropout placement:
      - BatchNorm1d after the first FC layer stabilises the distribution of
        hidden activations and allows higher learning rates.
      - CustomDropout (p=0.5 by default) after each BN+ReLU block prevents
        co-adaptation; placed in the FC layers where neurons are densely
        connected and overfitting risk is highest.
      - No dropout in conv blocks: spatial feature detectors benefit more
        from BN alone; dropout in early conv layers can destroy spatial
        structure needed for downstream tasks.
    """

    def __init__(self, num_classes: int = 37, dropout_p: float = 0.5):
        super().__init__()
        # VGG11 bottleneck: 512 * 7 * 7 = 25088
        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(dropout_p),
            nn.Linear(4096, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        return self.head(x)


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Classifier model.
        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.head = ClassificationHead(num_classes=num_classes, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x, return_features=False)   # [B, 512, 7, 7]
        return self.head(features)
