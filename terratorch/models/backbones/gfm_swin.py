# reference implementation https://github.com/mmendiet/GFM
# "Towards Geospatial Foundation Models via Continual Pretraining" (arXiv:2302.04476)

import logging
import sys
import types
from collections import OrderedDict

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from terratorch.datasets.utils import HLSBands, generate_bands_intervals
from terratorch.models.backbones.prithvi_swin import convert_weights_swin2mmseg
from terratorch.models.backbones.select_patch_embed_weights import select_patch_embed_weights
from terratorch.models.backbones.swin_encoder_decoder import MMSegSwinTransformer
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

logger = logging.getLogger(__name__)

# GFM was pretrained on GeoPile, which is RGB only.
GFM_PRETRAINED_BANDS = [HLSBands.RED, HLSBands.GREEN, HLSBands.BLUE]

GFM_HF = {"repo_id": "torchgeo/gfm", "filename": "gfm.pth"}

GFM_SWIN_BASE_ARGS = {
    "pretrain_img_size": 192,
    "patch_size": 4,
    "window_size": 6,
    "embed_dim": 128,
    "depths": (2, 2, 18, 2),
    "num_heads": (4, 8, 16, 32),
}

GFM_OUT_CHANNELS = [128, 256, 512, 1024]

# The upstream checkpoint carries SimMIM pretraining artifacts and a teacher branch
# that have no counterpart in MMSegSwinTransformer.
_ENCODER_PREFIX = "encoder."

# mmseg's Swin builds a norm per stage, which the upstream checkpoint does not carry, and a
# classifier head, which an encoder never uses. nn.LayerNorm default-initializes to weight=1
# and bias=0, i.e. exactly the identity, so the absent stage norms are deterministic rather
# than randomly initialized. Verified: models built under different seeds produce bitwise
# identical features.
_ALLOWED_MISSING_SUFFIXES = ("norm.weight", "norm.bias")
_ALLOWED_MISSING_KEYS = {"head.fc.weight", "head.fc.bias"}


def _install_yacs_stub() -> None:
    """Make the GFM checkpoint unpicklable without depending on `yacs`.

    The upstream checkpoint stores its training config as a pickled `yacs.config.CfgNode`,
    so `torch.load` needs that symbol to be importable. `yacs` is not a TerraTorch
    dependency and should not become one, since the config is discarded anyway. A stub is
    registered only when `yacs` is genuinely absent, via `setdefault`, so a real
    installation is never shadowed.
    """
    if "yacs" in sys.modules or "yacs.config" in sys.modules:
        return

    try:
        # Deliberately lazy: this probes whether a real yacs is installed.
        import yacs.config  # noqa: F401, PLC0415
    except ImportError:
        pass
    else:
        return

    yacs_module = types.ModuleType("yacs")
    config_module = types.ModuleType("yacs.config")

    class CfgNode(dict):
        """Minimal stand-in for `yacs.config.CfgNode`, sufficient to unpickle and discard."""

    config_module.CfgNode = CfgNode
    yacs_module.config = config_module

    sys.modules.setdefault("yacs", yacs_module)
    sys.modules.setdefault("yacs.config", config_module)


def _load_gfm_checkpoint(ckpt_data: str) -> dict[str, torch.Tensor]:
    """Load the raw upstream GFM checkpoint and return its `model` state dict.

    Args:
        ckpt_data (str): Local path, or an `https://hf.co/<repo>/resolve/<rev>/<file>` URL.

    Returns:
        The `model` entry, discarding optimizer / lr_scheduler / config / amp state.
    """
    if ckpt_data.find("https://hf.co/") > -1:
        repo_id = ckpt_data.split("/resolve/", maxsplit=1)[0].replace("https://hf.co/", "")
        filename = ckpt_data.rsplit("/", maxsplit=1)[-1]
        ckpt_data = hf_hub_download(repo_id=repo_id, filename=filename)

    _install_yacs_stub()

    # weights_only=True cannot read this checkpoint: it stores a pickled yacs CfgNode.
    checkpoint = torch.load(ckpt_data, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        msg = (
            f"Expected a GFM checkpoint containing a 'model' key, but got "
            f"{sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__}."
        )
        raise ValueError(msg)

    return checkpoint["model"]


def gfm_checkpoint_filter_fn(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
    pretrained_bands: list,
    model_bands: list,
) -> OrderedDict:
    """Convert an upstream GFM state dict into MMSegSwinTransformer naming.

    The upstream `model` dict holds four branches: `encoder.*` (the student, which is what we
    want), `teacher.*` (the EMA branch), and `decoder.*` / `projector.*` (pretraining heads).

    Args:
        state_dict: The upstream `model` state dict, or a full checkpoint containing `model`.
        model: The target model, used for patch embed band selection.
        pretrained_bands: Bands the checkpoint was pretrained on.
        model_bands: Bands the model is being built for.

    Returns:
        A state dict using mmseg key names, restricted to the encoder.
    """
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Keep the student encoder; drop teacher/decoder/projector.
    encoder_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not k.startswith(_ENCODER_PREFIX):
            continue
        stripped = k[len(_ENCODER_PREFIX) :]
        # mask_token is a SimMIM pretraining artifact; attn_mask buffers are non-persistent.
        if stripped == "mask_token" or stripped.endswith("attn_mask"):
            continue
        encoder_state_dict[stripped] = v

    # Renames Microsoft-Swin names to mmseg names, and critically permutes
    # downsample.reduction / downsample.norm to mmseg's PatchMerging unfold order.
    # Skipping that permutation loads without error but silently yields wrong features.
    converted = convert_weights_swin2mmseg(encoder_state_dict)

    # GFM's single final norm has no mmseg counterpart (mmseg norms per stage instead).
    for key in ("norm.weight", "norm.bias"):
        converted.pop(key, None)

    logger.info(f"GFM checkpoint: kept {len(converted)} encoder tensors.")
    logger.info(
        "The per-stage norms are absent from the GFM checkpoint and initialize to the "
        "identity (LayerNorm weight=1, bias=0), so features are deterministic."
    )

    return select_patch_embed_weights(converted, model, pretrained_bands, model_bands)


class GFMSwinEncoderWrapper(nn.Module):
    """Wraps MMSegSwinTransformer to emit a standard BCHW feature pyramid.

    `MMSegSwinTransformer.forward_features` returns only the last stage's tuple, because
    `stages` is an `nn.Sequential`, so this wrapper iterates the stages itself to collect
    every level. Each stage returns `(normed_NHWC, hw_shape, downsampled, down_hw_shape)`
    and consumes a single `(tokens, hw_shape)` tuple.

    Unlike `satlas_swin_*` and `prithvi_swin_*`, the output is already NCHW, so no
    `PermuteDims` neck is needed to feed decoders such as `UperNetDecoder`.
    """

    def __init__(self, swin_model: MMSegSwinTransformer, out_indices: tuple | list = (0, 1, 2, 3)) -> None:
        """
        Args:
            swin_model (MMSegSwinTransformer): The backbone module to be wrapped.
            out_indices (tuple | list): Indices of the stages whose features are returned.
        """
        super().__init__()
        self.model = swin_model
        self.out_indices = tuple(out_indices)

        # EncoderDecoderFactory requires out_channels, and infers patch_size by scanning
        # submodules -- MMSegSwinTransformer's patch_embed does not expose one.
        self.out_channels = [c for i, c in enumerate(GFM_OUT_CHANNELS) if i in self.out_indices]
        self.patch_size = GFM_SWIN_BASE_ARGS["patch_size"]

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:  # noqa: ARG002
        outs = []
        stage_input = self.model.patch_embed(x)
        for i, stage in enumerate(self.model.stages):
            normed, _, downsampled, down_hw_shape = stage(stage_input)
            if i in self.out_indices:
                outs.append(normed.permute(0, 3, 1, 2).contiguous())
            stage_input = (downsampled, down_hw_shape)

        return outs


@TERRATORCH_BACKBONE_REGISTRY.register
def gfm_swin_base(
    model_bands: list | None = None,
    pretrained: bool = False,  # noqa: FBT001, FBT002
    ckpt_data: str | None = None,
    out_indices: tuple | list = (0, 1, 2, 3),
    **kwargs,
) -> GFMSwinEncoderWrapper:
    """GFM Swin-B encoder, pretrained on GeoPile via continual pretraining.

    Reference: "Towards Geospatial Foundation Models via Continual Pretraining"
    (arXiv:2302.04476), https://github.com/mmendiet/GFM. Weights are Apache-2.0 and
    distributed as `torchgeo/gfm` on the Hugging Face Hub.

    Returns a four-level BCHW feature pyramid with channels [128, 256, 512, 1024] at strides
    [4, 8, 16, 32]. Because the output is already NCHW, this backbone needs no `PermuteDims`
    neck in `EncoderDecoderFactory`.

    Preprocessing: scale inputs to [0, 1], then normalize with ImageNet statistics,
    mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

    Pretraining used 192x192 RGB imagery. There is no absolute position embedding, so larger
    input sizes work without interpolation. Requesting bands beyond RGB is supported and
    follows the usual TerraTorch convention: pretrained RGB weights are placed at the matching
    band positions and any additional band is randomly initialized, which reinterprets an
    RGB-only patch embedding.

    The checkpoint carries no per-stage norms. Those initialize to the identity
    (LayerNorm weight=1, bias=0), so results do not depend on the random seed.

    Args:
        model_bands (list): Bands the model is built for. Defaults to RGB.
        pretrained (bool): Whether to load the pretrained weights.
        ckpt_data (str | None): Path or hf.co URL of a checkpoint, taking precedence over the
            Hugging Face download.
        out_indices (tuple | list): Indices of the stages whose features are returned.

    Returns:
        GFMSwinEncoderWrapper
    """
    if model_bands is None:
        model_bands = GFM_PRETRAINED_BANDS
    model_bands = [HLSBands.try_convert_to_hls_bands_enum(b) for b in model_bands]
    model_bands = generate_bands_intervals(model_bands)

    kwargs["in_chans"] = len(model_bands)

    model = MMSegSwinTransformer(**GFM_SWIN_BASE_ARGS, **kwargs)

    if pretrained or ckpt_data is not None:
        if ckpt_data is None:
            ckpt_data = hf_hub_download(**GFM_HF)
        checkpoint = _load_gfm_checkpoint(ckpt_data)
        checkpoint = gfm_checkpoint_filter_fn(checkpoint, model, GFM_PRETRAINED_BANDS, model_bands)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        if unexpected_keys:
            msg = f"Unexpected keys when loading the GFM checkpoint: {sorted(unexpected_keys)}"
            raise ValueError(msg)

        unaccounted = [
            k for k in missing_keys if k not in _ALLOWED_MISSING_KEYS and not k.endswith(_ALLOWED_MISSING_SUFFIXES)
        ]
        if unaccounted:
            msg = f"Missing keys when loading the GFM checkpoint: {sorted(unaccounted)}"
            raise ValueError(msg)

    encoder = GFMSwinEncoderWrapper(model, out_indices=out_indices)
    encoder.model_bands = model_bands
    encoder.pretrained_bands = GFM_PRETRAINED_BANDS

    return encoder
