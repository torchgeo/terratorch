import os
import sys
import textwrap

import pytest
import rasterio
import torch

from terratorch.models.backbones import universat as universat_module
from terratorch.models.backbones.universat import UniverSatEncoderWrapper
from terratorch.models.encoder_decoder_factory import EncoderDecoderFactory
from terratorch.registry import BACKBONE_REGISTRY
from terratorch.tasks.embedding_generation import EmbeddingGenerationTask

EMBED_DIM = 768

# Minimal stand-ins for the official UniverSat hubconf.py and modality_registry.py: same entry points, `encode`
# signature and registry tables. The fake model pools every modality to the output grid and sums them. Like the real
# hub module, it holds its weights under `.model`, so its state dict keys start with "model.".
FAKE_REGISTRY = textwrap.dedent(
    """
    INPUT_RES = {"s2": 10, "s2l1c": 10, "s1": 10, "alos": 30, "naip": 1.25}
    SUBPATCHES = {"s2": 1, "s2l1c": 1, "s1": 1, "alos": 1, "naip": 10}
    WAVELENGTHS = {
        "s2": [0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.833, 0.865, 1.61, 2.19],
        "s2l1c": [0.442] * 13,
        "s1": ["VV", "VH", "Ratio_VV_VH"],
        "alos": ["HH", "HV", "Ratio_HH_HV"],
        "naip": [0.665, 0.56, 0.49, 0.81],
    }
    """
)

FAKE_HUBCONF = textwrap.dedent(
    """
    import os
    import sys

    import torch
    from torch import nn

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import modality_registry

    dependencies = ["torch"]


    class _FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(768))


    class _FakeUniverSat(nn.Module):
        embed_dim = 768

        def __init__(self):
            super().__init__()
            self.model = _FakeEncoder()
            self.calls = []

        def encode(self, x, patch_size=40, latent_grid=None, output_grid=None, wavelengths=None, input_res=None,
                   subpatches=None, **forward_kwargs):
            images = {k: v for k, v in x.items() if k in modality_registry.WAVELENGTHS or k in (wavelengths or {})}
            self.calls.append(dict(keys=sorted(x), patch_size=patch_size, latent_grid=latent_grid,
                                   output_grid=output_grid, wavelengths=wavelengths, input_res=input_res,
                                   subpatches=subpatches, forward_kwargs=forward_kwargs, inputs=dict(x)))
            name, first = next(iter(images.items()))
            res = (input_res or {}).get(name, modality_registry.INPUT_RES.get(name))
            side = first.shape[-1] // max(int(patch_size / res), 1)
            sub = max((subpatches or {}).get(k, modality_registry.SUBPATCHES.get(k, 1)) for k in images)
            grid = output_grid or side * sub
            tokens = 0
            for image in images.values():
                image = image.mean(1) if image.ndim == 5 else image
                pooled = torch.nn.functional.adaptive_avg_pool2d(image, grid)
                tokens = tokens + pooled.flatten(2).transpose(1, 2).mean(-1, keepdim=True)
            return tokens * self.model.scale, {}


    def from_pretrained(repo_id="g-astruc/UniverSat", compile=False, **kwargs):
        model = _FakeUniverSat()
        model.pretrained, model.repo_id, model.compile_mode, model.hub_kwargs = True, repo_id, compile, kwargs
        return model


    def universat(pretrained=False, size="base", modalities_dict=None, compile="max", **kwargs):
        model = _FakeUniverSat()
        model.pretrained, model.compile_mode, model.modalities_dict = pretrained, compile, modalities_dict
        return model
    """
)


@pytest.fixture
def hub_dir(tmp_path):
    (tmp_path / "hubconf.py").write_text(FAKE_HUBCONF)
    (tmp_path / "modality_registry.py").write_text(FAKE_REGISTRY)
    yield str(tmp_path)
    # The wrapper reads the registry of the loaded UniverSat code: do not leak it to other tests.
    sys.modules.pop("modality_registry", None)
    while str(tmp_path) in sys.path:
        sys.path.remove(str(tmp_path))


def build(hub_dir, **kwargs):
    return BACKBONE_REGISTRY.build("universat", hub_repo=hub_dir, hub_source="local", **kwargs)


def test_universat_default_grid(hub_dir):
    assert build(hub_dir).model.compile_mode == "fast"
    backbone = build(hub_dir, compile_model="max", hub_kwargs={"revision": "main"})
    assert isinstance(backbone, UniverSatEncoderWrapper)
    assert backbone.out_channels == [EMBED_DIM]
    assert backbone.model.pretrained
    assert backbone.model.compile_mode == "max"
    assert backbone.model.hub_kwargs == {"revision": "main"}

    out = backbone(torch.rand(2, 10, 64, 64))
    assert isinstance(out, list)
    # Dense default, one token per sub-patch: 64 px at 10 m with 40 m patches and sub-patch factor 1 gives 16 x 16.
    assert out[0].shape == (2, 16 * 16, EMBED_DIM)
    call = backbone.model.calls[-1]
    assert call["keys"] == ["s2"]
    assert call["patch_size"] == 40
    assert call["latent_grid"] is None
    assert call["output_grid"] == 16
    # Registry metadata is left to UniverSat unless overridden.
    assert call["wavelengths"] is None
    assert call["input_res"] is None
    assert call["subpatches"] is None


def test_universat_output_resolution(hub_dir):
    image = torch.rand(1, 10, 64, 64)
    assert build(hub_dir, output_res=10)(image)[0].shape == (1, 64 * 64, EMBED_DIM)
    assert build(hub_dir, output_res=20)(image)[0].shape == (1, 32 * 32, EMBED_DIM)
    assert build(hub_dir, output_grid=8, output_res=10)(image)[0].shape == (1, 8 * 8, EMBED_DIM)
    backbone = build(hub_dir, patch_size_meters=80, latent_grid=64, encode_kwargs={"keep_subpatch": True})
    assert backbone(image)[0].shape == (1, 8 * 8, EMBED_DIM)
    call = backbone.model.calls[-1]
    assert (call["patch_size"], call["latent_grid"], call["forward_kwargs"]) == (80, 64, {"keep_subpatch": True})
    # Dense default with a very high resolution input: one token per sub-patch (4 m), not per pixel (1.25 m).
    out = build(hub_dir)({"s2": torch.rand(1, 10, 64, 64), "naip": torch.rand(1, 4, 512, 512)})
    assert out[0].shape == (1, 160 * 160, EMBED_DIM)


def test_universat_multimodal_mixed_resolution(hub_dir):
    backbone = build(hub_dir, output_res=10, wavelengths={"S1GRD": ["VV", "VH"]})
    batch = {
        "S2L2A": torch.rand(2, 10, 64, 64),  # 640 m at 10 m
        "S1GRD": torch.rand(2, 2, 64, 64),  # 640 m at 10 m, VV and VH only (wavelengths override)
        "naip": torch.rand(2, 4, 512, 512),  # 640 m at 1.25 m
    }
    out = backbone(batch)
    assert out[0].shape == (2, 64 * 64, EMBED_DIM)
    call = backbone.model.calls[-1]
    assert call["keys"] == ["naip", "s1", "s2"]  # all modalities encoded in one call
    assert call["output_grid"] == 64
    assert call["wavelengths"] == {"s1": ["VV", "VH"]}  # overrides only for their own modality

    with pytest.raises(ValueError, match="same tile"):
        backbone({"s2": torch.rand(1, 10, 64, 64), "naip": torch.rand(1, 4, 256, 256)})
    with pytest.raises(ValueError, match="batch size"):
        backbone({"s2": torch.rand(1, 10, 64, 64), "s1": torch.rand(2, 3, 64, 64)})
    with pytest.raises(ValueError, match="square"):
        backbone(torch.rand(1, 10, 32, 16))


def test_universat_unknown_and_unseen_sensors(hub_dir):
    backbone = build(hub_dir)
    with pytest.raises(ValueError, match="Unknown UniverSat modalities"):
        backbone({"s2": torch.rand(1, 10, 32, 32), "S2": torch.rand(1, 10, 32, 32)})

    # A sensor outside the registry, described by its metadata.
    unseen = {"wavelengths": {"superspectral": [0.45, 0.55, 0.65, 0.85, 1.6]}, "input_res": {"superspectral": 5.0}}
    backbone = build(hub_dir, output_res=10, **unseen)
    out = backbone({"superspectral": torch.rand(1, 5, 128, 128), "s2": torch.rand(1, 10, 64, 64)})
    assert out[0].shape == (1, 64 * 64, EMBED_DIM)
    assert backbone.model.calls[-1]["input_res"] == {"superspectral": 5.0}
    with pytest.raises(ValueError, match="input_res"):
        build(hub_dir, output_res=10, wavelengths=unseen["wavelengths"])({"superspectral": torch.rand(1, 5, 8, 8)})


def test_universat_ratio_bands(hub_dir):
    backbone = build(hub_dir)
    s1 = torch.rand(2, 2, 3, 8, 8) + 0.5  # VV, VH time series
    alos = torch.rand(2, 2, 8, 8) + 0.5  # HH, HV
    backbone({"s1": s1, "s1_dates": torch.ones(2, 3), "alos": alos})
    inputs = backbone.model.calls[-1]["inputs"]
    assert inputs["s1"].shape == (2, 3, 3, 8, 8)  # (B, T, C, H, W) with the ratio band
    assert torch.allclose(inputs["s1"][:, :, 2], s1[:, 0] / (s1[:, 1] + 1e-10))
    assert torch.allclose(inputs["alos"][:, 2], alos[:, 0] / (alos[:, 1] + 1e-10))

    # Clipping as in UniverSat's data loaders.
    backbone({"s1": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(1, 2, 4, 4)})
    assert backbone.model.calls[-1]["inputs"]["s1"][:, 2].max() == 1e4

    # Left unchanged if already present or if the band list is overridden.
    backbone({"s1": torch.rand(1, 3, 8, 8)})
    assert backbone.model.calls[-1]["inputs"]["s1"].shape[1] == 3
    overridden = build(hub_dir, wavelengths={"s1": ["VV", "VH"]})
    overridden({"s1": torch.rand(1, 2, 8, 8)})
    assert overridden.model.calls[-1]["inputs"]["s1"].shape[1] == 2
    with pytest.raises(ValueError, match="got 2 channels"):  # without the ratio, 2 channels do not match s1
        build(hub_dir, add_ratio_bands=False)({"s1": torch.rand(1, 2, 8, 8)})

    # Standardization applies after the ratio band is added, with statistics for all three bands.
    backbone = build(hub_dir, mean={"s1": [0.0, 0.0, 1.0]}, std={"s1": [1.0, 1.0, 2.0]})
    backbone({"s1": torch.full((1, 2, 4, 4), 3.0)})
    assert torch.allclose(backbone.model.calls[-1]["inputs"]["s1"][:, 2], torch.zeros(1, 4, 4), atol=1e-6)


def test_universat_standardization(hub_dir):
    mean = [float(i) for i in range(10)]
    backbone = build(hub_dir, mean={"S2L2A": mean, "s1": [1.0] * 3}, std={"S2L2A": [2.0] * 10, "s1": [1.0] * 3})
    s2 = torch.tensor(mean)[None, :, None, None].expand(1, 10, 8, 8).clone() + 2.0
    s1 = torch.full((1, 3, 8, 8), 5.0)
    backbone({"s2": s2, "s1": s1})
    inputs = backbone.model.calls[-1]["inputs"]
    assert torch.allclose(inputs["s2"], torch.ones_like(s2))
    assert torch.allclose(inputs["s1"], torch.full_like(s1, 4.0))
    # Standard deviations are floored, as in UniverSat.
    backbone = build(hub_dir, mean={"s2": [0.0] * 10}, std={"s2": [0.0] * 10})
    backbone(torch.ones(1, 10, 4, 4))
    assert torch.isfinite(backbone.model.calls[-1]["inputs"]["s2"]).all()
    # 2-band S1 statistics: the ratio band is added first, so the statistics must cover it.
    backbone = build(hub_dir, mean={"s1": [1.0] * 2}, std={"s1": [1.0] * 2})
    with pytest.raises(ValueError, match=r"mean/std have 2 values, but the input has 3 channels.*Ratio_VV_VH"):
        backbone({"s1": torch.rand(1, 2, 8, 8)})
    with pytest.raises(ValueError, match="together"):
        build(hub_dir, mean={"s2": mean})
    with pytest.raises(ValueError, match="same modalities"):
        build(hub_dir, mean={"s2": mean}, std={"s1": [1.0] * 3})


def test_universat_time_series(hub_dir):
    backbone = build(hub_dir, output_res=10)
    s2 = torch.rand(2, 10, 5, 32, 32)  # TerraTorch layout (B, C, T, H, W)
    s1 = torch.rand(2, 3, 7, 32, 32)
    s2_doy = torch.tensor([[10, 40, 70, 100, 130], [5, 35, 65, 95, 125]])
    s1_doy = torch.arange(7).expand(2, -1) * 12 + 1
    out = backbone({"S2L2A": s2, "S2L2A_dates": s2_doy, "S1GRD": s1, "S1GRD_dates": s1_doy})
    assert out[0].shape == (2, 32 * 32, EMBED_DIM)
    inputs = backbone.model.calls[-1]["inputs"]
    assert sorted(inputs) == ["s1", "s1_dates", "s2", "s2_dates"]
    assert inputs["s2"].shape == (2, 5, 10, 32, 32)  # UniverSat layout (B, T, C, H, W)
    assert inputs["s1"].shape == (2, 7, 3, 32, 32)
    assert torch.equal(inputs["s2_dates"], s2_doy)

    # Dates as an argument; a single image with dates is a one-date series; mixed static and temporal inputs.
    backbone(s2, dates=s2_doy)
    backbone({"s2": torch.rand(2, 10, 32, 32), "s1": s1}, dates={"S1GRD": s1_doy, "s2": s2_doy[:, :1]})
    assert backbone.model.calls[-1]["inputs"]["s2"].shape == (2, 1, 10, 32, 32)
    backbone({"s2": torch.rand(2, 10, 32, 32), "s1": s1, "s1_dates": s1_doy})
    assert backbone.model.calls[-1]["inputs"]["s2"].shape == (2, 10, 32, 32)

    with pytest.raises(ValueError, match="dates of shape"):
        backbone(s2)
    with pytest.raises(ValueError, match="without images"):
        backbone({"s2": s2, "s2_dates": s2_doy, "s1_dates": s1_doy})


def test_universat_untrained_and_checkpoint(hub_dir, tmp_path):
    backbone = build(hub_dir, pretrained=False, hub_kwargs={"modalities_dict": {"pastishd": ["s2"]}})
    assert not backbone.model.pretrained
    assert backbone.model.modalities_dict == {"pastishd": ["s2"]}

    # State dicts of the inner encoder, the hub module, the wrapper, a TerraTorch task and a Lightning training run.
    weights = torch.full((EMBED_DIM,), 3.0)
    for i, prefix in enumerate(["", "model.", "model.model.", "model.encoder.model.model.", "model.encoder."]):
        ckpt = tmp_path / f"universat_{i}.pt"
        torch.save({"state_dict": {f"{prefix}scale": weights}} if i == 3 else {f"{prefix}scale": weights}, ckpt)
        backbone = build(hub_dir, pretrained=False, ckpt_path=str(ckpt))
        assert torch.equal(backbone.model.model.scale, weights), prefix

    # Lightning checkpoints hold non-tensor objects, loadable with ckpt_weights_only=False.
    ckpt = tmp_path / "lightning.ckpt"
    torch.save({"state_dict": {"model.encoder.scale": weights}, "hyper_parameters": {"cfg": object}}, ckpt)
    with pytest.raises(Exception, match="Weights only load failed"):
        build(hub_dir, pretrained=False, ckpt_path=str(ckpt))
    backbone = build(hub_dir, pretrained=False, ckpt_path=str(ckpt), ckpt_weights_only=False)
    assert torch.equal(backbone.model.model.scale, weights)

    ckpt = tmp_path / "other.pt"
    torch.save({"decoder.weight": weights}, ckpt)
    with pytest.raises(ValueError, match="No key"):
        build(hub_dir, pretrained=False, ckpt_path=str(ckpt))


@torch.no_grad()
def test_universat_embedding_generation_tiff(hub_dir, tmp_path):
    out_dir = tmp_path / "embeddings"
    task = EmbeddingGenerationTask(
        model_args={
            "backbone": "universat",
            "backbone_hub_repo": hub_dir,
            "backbone_hub_source": "local",
            "backbone_output_res": 10,
        },
        output_dir=str(out_dir),
        output_format="tiff",
    )
    batch = {
        "image": torch.rand(2, 10, 16, 16),
        "filename": ["a.tif", "b.tif"],
        "metadata": {
            "crs": ["EPSG:32633", "EPSG:32633"],
            "bounds": torch.tensor([[500000.0, 5000000.0, 500160.0, 5000160.0]] * 2),
        },
    }
    task.predict_step(batch)

    with rasterio.open(out_dir / "layer_00" / "a_embedding.tif") as src:
        assert src.count == EMBED_DIM
        assert (src.height, src.width) == (16, 16)
        assert src.res == (10.0, 10.0)


def test_universat_alias_band_layouts(hub_dir):
    # 2-band Sentinel-1 (TerraTorch S1GRD / S1RTC): the VV/VH ratio band is added.
    backbone = build(hub_dir)
    backbone({"S1RTC": torch.rand(1, 2, 8, 8)})
    assert backbone.model.calls[-1]["inputs"]["s1"].shape[1] == 3

    # 12-band TerraTorch S2L2A: B01 and B09 are dropped, statistics for 12 or for 10 bands.
    s2 = torch.arange(12.0)[None, :, None, None].expand(1, 12, 8, 8)
    backbone({"S2L2A": s2})
    kept = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
    assert torch.equal(backbone.model.calls[-1]["inputs"]["s2"][0, :, 0, 0], torch.tensor(kept, dtype=torch.float))
    mean12, std12 = [float(i) for i in range(12)], [1.0] * 12
    backbone = build(hub_dir, mean={"S2L2A": mean12}, std={"S2L2A": std12})
    backbone({"S2L2A": s2})
    assert backbone.model.calls[-1]["inputs"]["s2"].abs().max() == 0
    backbone = build(hub_dir, mean={"S2L2A": [float(i) for i in kept]}, std={"S2L2A": [1.0] * 10})
    backbone({"S2L2A": s2, "S2L2A_dates": torch.ones(1, 1)})
    assert backbone.model.calls[-1]["inputs"]["s2"].abs().max() == 0

    # A wavelengths override disables the selection: all 12 bands are encoded.
    wavelengths = [0.442, 0.49, 0.56, 0.665, 0.705, 0.74, 0.783, 0.833, 0.865, 0.945, 1.61, 2.19]
    backbone = build(hub_dir, wavelengths={"S2L2A": wavelengths})
    backbone({"S2L2A": s2})
    assert backbone.model.calls[-1]["inputs"]["s2"].shape[1] == 12


def test_universat_channel_validation(hub_dir):
    backbone = build(hub_dir)
    with pytest.raises(ValueError, match="got 12 channels"):
        backbone({"s2": torch.rand(1, 12, 8, 8)})
    with pytest.raises(ValueError, match="got 4 channels"):
        backbone({"s1": torch.rand(1, 4, 8, 8)})
    with pytest.raises(ValueError, match="got 3 channels"):
        build(hub_dir, wavelengths={"x": [0.5, 0.6]}, input_res={"x": 10})({"x": torch.rand(1, 3, 8, 8)})


def test_universat_encode_kwargs_and_output_res(hub_dir):
    with pytest.raises(ValueError, match="repeats arguments"):
        build(hub_dir, encode_kwargs={"output_grid": 4})
    with pytest.warns(UserWarning, match="does not divide"):
        out = build(hub_dir, output_res=20)(torch.rand(1, 10, 15, 15))  # 150 m tile: 8 tokens of 18.75 m
    assert out[0].shape == (1, 8 * 8, EMBED_DIM)


def test_universat_hub_loading_defaults(monkeypatch):
    calls = []

    def fake_load(repo, entrypoint, **kwargs):
        calls.append((repo, entrypoint, kwargs))
        msg = "stop"
        raise RuntimeError(msg)

    monkeypatch.setattr(universat_module.torch.hub, "load", fake_load)
    for hub_repo in (universat_module.UNIVERSAT_HUB_REPO, "gastruc/UniverSat:main"):
        with pytest.raises(RuntimeError, match="stop"):
            BACKBONE_REGISTRY.build("universat", hub_repo=hub_repo)
    (_, entrypoint, pinned), (_, _, branch) = calls
    assert entrypoint == "from_pretrained"
    # No interactive trust prompt by default, so batch jobs work; pinned commits skip torch.hub's ref validation.
    assert pinned["trust_repo"] is True and pinned["skip_validation"] is True
    assert pinned["compile"] == "fast"
    assert branch["skip_validation"] is False


def test_universat_segmentation(hub_dir):
    model = EncoderDecoderFactory().build_model(
        task="segmentation",
        backbone="universat",
        backbone_hub_repo=hub_dir,
        backbone_hub_source="local",
        backbone_output_res=10,
        necks=[{"name": "ReshapeTokensToImage", "remove_cls_token": False}],
        decoder="FCNDecoder",
        num_classes=3,
    )
    batch = {"S2L2A": torch.rand(2, 10, 3, 32, 32), "S2L2A_dates": torch.tensor([[10, 40, 70], [5, 35, 65]])}
    out = model(batch).output
    assert out.shape == (2, 3, 32, 32)
    out.sum().backward()


@pytest.mark.slow
@pytest.mark.skipif("UNIVERSAT_HUB_DIR" not in os.environ, reason="needs a UniverSat checkout and weights")
def test_universat_real_model():
    backbone = BACKBONE_REGISTRY.build(
        "universat", hub_repo=os.environ["UNIVERSAT_HUB_DIR"], hub_source="local", compile_model=False
    ).eval()
    with torch.no_grad():
        tokens = backbone({"S2L2A": torch.randn(1, 12, 32, 32), "S1GRD": torch.rand(1, 2, 32, 32)})[0]
    assert tokens.shape == (1, 8 * 8, EMBED_DIM)
    assert torch.isfinite(tokens).all() and tokens.abs().sum() > 0
