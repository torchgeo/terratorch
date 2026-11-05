import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY

from .utils import ConvModule


# Adapted from MMSegmentation
@TERRATORCH_DECODER_REGISTRY.register
class UperNetDecoder(nn.Module):
    """UperNetDecoder. Adapted from MMSegmentation."""

    def __init__(
        self,
        embed_dim: list[int],
        pool_scales: tuple[int] = (1, 2, 3, 6),
        channels: int = 256,
        align_corners: bool = True,  # noqa: FBT001, FBT002
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            pool_scales (tuple[int], optional): Pooling scales used in Pooling Pyramid
                Module applied on the last feature. Default: (1, 2, 3, 6).
            channels (int, optional): Channels used in the decoder. Defaults to 256.
            align_corners (bool, optional): Wheter to align corners in rescaling. Defaults to True.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.out_channels = channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.embed_dim[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, inplace=True
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for embed_dim in self.embed_dim[:-1]:  # skip the top layer
            l_conv = ConvModule(
                embed_dim,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.embed_dim) * self.channels, self.channels, 3, padding=1, inplace=True)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """

        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        """Constructor

        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
                Module.
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_seg.
            align_corners (bool): align_corners argument of F.interpolate.
        """
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs
