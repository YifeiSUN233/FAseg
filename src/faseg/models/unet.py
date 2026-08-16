"""
U-Net architectures for 2D TOF ultrasound segmentation.

Provides both a 3-level (:class:`UNet`) and a deeper 4-level
(:class:`UNet2`) variant with optional BatchNorm and Dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two sequential 3×3 conv layers, each followed by (optional) BatchNorm
    and ReLU.  Optionally appends Dropout2d at the end."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout_prob: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch) if use_bn else nn.Identity(),
            nn.ReLU(inplace=True),
        ]
        if dropout_prob > 0:
            layers.append(nn.Dropout2d(dropout_prob))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


# ---------------------------------------------------------------------------
# U-Net (3-level) — primary architecture
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """3-level U-Net with symmetric encoder/decoder and skip connections.

    Parameters
    ----------
    in_ch : int
        Number of input channels (default 1 for grayscale).
    base_ch : int
        Channels in the first encoder layer; doubled at each down-sampling.
    num_classes : int
        Number of output classes (default 2 for binary segmentation).
    dropout_prob : float
        Dropout2d probability applied after each DoubleConv.
    use_bn : bool
        If True, insert BatchNorm2d after every convolution.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_classes: int = 2,
        dropout_prob: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch, dropout_prob, use_bn)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2, dropout_prob, use_bn)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4, dropout_prob, use_bn)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8, dropout_prob, use_bn)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, dropout_prob, use_bn)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, dropout_prob, use_bn)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, dropout_prob, use_bn)

        # Classifier
        self.classifier = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape ``(B, in_ch, H, W)``

        Returns
        -------
        Tensor, shape ``(B, num_classes, H, W)`` — raw logits.
        """
        _, _, H0, W0 = x.size()

        # Pad to multiple of 8 for clean down/up-sampling
        pad_h = (8 - H0 % 8) % 8
        pad_w = (8 - W0 % 8) % 8
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.classifier(d1)
        # Crop back to original spatial size
        out = out[:, :, pad_top:pad_top + H0, pad_left:pad_left + W0]
        return out


# ---------------------------------------------------------------------------
# U-Net2 (4-level) — deeper variant
# ---------------------------------------------------------------------------

class UNet2(nn.Module):
    """4-level U-Net — one level deeper than :class:`UNet`.

    Same interface as :class:`UNet` but with an additional encoder/decoder
    level.  Padding is aligned to multiples of 16.
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_classes: int = 2,
        dropout_prob: float = 0.0,
        use_bn: bool = False,
    ):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch, dropout_prob, use_bn)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base_ch, base_ch * 2, dropout_prob, use_bn)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4, dropout_prob, use_bn)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(base_ch * 4, base_ch * 8, dropout_prob, use_bn)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 8, base_ch * 16, dropout_prob, use_bn)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch * 16, base_ch * 8, dropout_prob, use_bn)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4, dropout_prob, use_bn)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2, dropout_prob, use_bn)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch, dropout_prob, use_bn)

        # Classifier
        self.classifier = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — pads to multiple of 16, then crops back."""
        _, _, H0, W0 = x.size()

        pad_h = (16 - H0 % 16) % 16
        pad_w = (16 - W0 % 16) % 16
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.classifier(d1)
        out = out[:, :, pad_top:pad_top + H0, pad_left:pad_left + W0]
        return out
