"""Segmentation model — U-Net style with VGG11 encoder
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _decoder_block(in_ch: int, skip_ch: int, out_ch: int) -> nn.Sequential:
    """TransposedConv upsample -> concat with skip (handled externally) -> conv block.

    The TransposedConv doubles spatial resolution (stride=2, kernel=2).
    After concatenation with the skip connection the combined channel count
    is (in_ch//2 + skip_ch) which is then reduced to out_ch by two 3x3 convs.
    """
    # TransposedConv: in_ch -> in_ch//2  (halve channels, double spatial)
    up_ch = in_ch // 2
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, up_ch, kernel_size=2, stride=2),
        nn.BatchNorm2d(up_ch),
        nn.ReLU(inplace=True),
    )


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two 3x3 conv -> BN -> ReLU applied after skip-connection concat."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Encoder  : VGG11 convolutional backbone (5 blocks, MaxPool between each).
    Bottleneck: 512-ch feature map at 7x7.
    Decoder  : 5 symmetric up-stages using TransposedConvolutions + skip connections.
    Output   : [B, num_classes, 224, 224] segmentation logits.

    Loss justification (see train.py):
      Combined Cross-Entropy + Dice loss.
      CE provides strong gradient signal per pixel; Dice directly optimises the
      overlap metric used for evaluation (DSC), especially on class-imbalanced
      segmentation tasks where background dominates.

    Dropout placement justification:
      CustomDropout is applied in the bottleneck only (between encoder and decoder).
      Applying it in early conv layers would destroy spatial structure needed for
      precise skip connections; applying it in every decoder block was empirically
      found to destabilise mask predictions.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the bottleneck dropout.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.bottleneck_drop = CustomDropout(dropout_p)

        # ------------------------------------------------------------------ #
        # Decoder — mirrors the 5-block encoder.
        # Spatial progression (with 224x224 input):
        #   Bottleneck: 512ch  7x7
        #   Up1:        512ch 14x14   skip f5 (512ch 14x14)  -> conv to 512
        #   Up2:        256ch 28x28   skip f4 (512ch 28x28)  -> conv to 256
        #   Up3:        128ch 56x56   skip f3 (256ch 56x56)  -> conv to 128
        #   Up4:         64ch 112x112 skip f2 (128ch 112x112)-> conv to 64
        #   Up5:         32ch 224x224 skip f1 (64ch  224x224)-> conv to 64
        # ------------------------------------------------------------------ #

        # Up-stage 1: 7x7 -> 14x14
        self.up1     = _decoder_block(512, 512, 512)   # TransConv: 512->256, then cat f5(512)->768
        self.conv1   = _conv_block(256 + 512, 512)

        # Up-stage 2: 14x14 -> 28x28
        self.up2     = _decoder_block(512, 512, 256)   # TransConv: 512->256, cat f4(512)->512
        self.conv2   = _conv_block(256 + 512, 256)

        # Up-stage 3: 28x28 -> 56x56
        self.up3     = _decoder_block(256, 256, 128)   # TransConv: 256->128, cat f3(256)->384
        self.conv3   = _conv_block(128 + 256, 128)

        # Up-stage 4: 56x56 -> 112x112
        self.up4     = _decoder_block(128, 128, 64)    # TransConv: 128->64, cat f2(128)->192
        self.conv4   = _conv_block(64 + 128, 64)

        # Up-stage 5: 112x112 -> 224x224
        self.up5     = _decoder_block(64, 64, 64)      # TransConv: 64->32, cat f1(64)->96
        self.conv5   = _conv_block(32 + 64, 64)

        # Final 1x1 conv to produce class logits
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.
        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        bottleneck, features = self.encoder(x, return_features=True)
        # bottleneck: [B, 512, 7, 7]
        # features: f1..f5 at respective spatial sizes

        z = self.bottleneck_drop(bottleneck)

        # Up 1
        z = self.up1(z)                                         # [B, 256, 14, 14]
        z = torch.cat([z, features["f5"]], dim=1)               # [B, 768, 14, 14]
        z = self.conv1(z)                                       # [B, 512, 14, 14]

        # Up 2
        z = self.up2(z)                                         # [B, 256, 28, 28]
        z = torch.cat([z, features["f4"]], dim=1)               # [B, 768, 28, 28] -- wait, 256+512=768
        z = self.conv2(z)                                       # [B, 256, 28, 28]

        # Up 3
        z = self.up3(z)                                         # [B, 128, 56, 56]
        z = torch.cat([z, features["f3"]], dim=1)               # [B, 384, 56, 56]
        z = self.conv3(z)                                       # [B, 128, 56, 56]

        # Up 4
        z = self.up4(z)                                         # [B, 64, 112, 112]
        z = torch.cat([z, features["f2"]], dim=1)               # [B, 192, 112, 112]
        z = self.conv4(z)                                       # [B, 64, 112, 112]

        # Up 5
        z = self.up5(z)                                         # [B, 32, 224, 224]
        z = torch.cat([z, features["f1"]], dim=1)               # [B, 96,  224, 224]
        z = self.conv5(z)                                       # [B, 64,  224, 224]

        return self.final_conv(z)                               # [B, num_classes, 224, 224]
