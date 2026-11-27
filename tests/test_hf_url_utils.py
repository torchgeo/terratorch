from unittest.mock import patch

import pytest

from terratorch.models.backbones.terramind.model import terramind_register as tr
from terratorch.utils import check_url_scheme, parse_hf_url


class TestParseHfUrl:
    def test_revision_and_filename(self):
        url = "https://hf.co/org/model/resolve/abc123/weights.pt"
        assert parse_hf_url(url) == ("org/model", "abc123", "weights.pt")

    def test_branch_revision(self):
        url = "https://hf.co/org/model/resolve/main/checkpoint.pth"
        assert parse_hf_url(url) == ("org/model", "main", "checkpoint.pth")

    def test_filename_in_subdirectory_is_kept(self):
        url = "https://hf.co/org/model/resolve/abc123/checkpoints/v2/weights.pt"
        assert parse_hf_url(url) == ("org/model", "abc123", "checkpoints/v2/weights.pt")

    def test_no_revision(self):
        url = "https://hf.co/org/model/resolve/weights.pt"
        assert parse_hf_url(url) == ("org/model", None, "weights.pt")

    def test_url_encoded_revision(self):
        url = "https://hf.co/org/model/resolve/refs%2Fpr%2F1/weights.pt"
        assert parse_hf_url(url) == ("org/model", "refs/pr/1", "weights.pt")


class TestCheckUrlScheme:
    @pytest.mark.parametrize("url", ["http://example.com/a.tif", "https://example.com/a.tif", "HTTPS://example.com/a"])
    def test_allowed(self, url):
        assert check_url_scheme(url) == url

    @pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/a", "/local/path", "data:text/plain,x"])
    def test_rejected(self, url):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            check_url_scheme(url)


class TestTerramindRevisionOverride:
    def _download_revision(self, **build_kwargs):
        captured = {}

        class StopAfterDownloadError(Exception):
            pass

        def fake_download(*, revision=None, **_):
            captured["revision"] = revision
            raise StopAfterDownloadError

        with patch.object(tr, "hf_hub_download", side_effect=fake_download):
            with pytest.raises(StopAfterDownloadError):
                tr.build_terrammind_vit(
                    variant="terramind_v1_tiny",
                    pretrained=True,
                    encoder_depth=1,
                    dim=16,
                    num_heads=1,
                    modalities=["untok_sen2l2a@224"],
                    **build_kwargs,
                )
        return captured["revision"]

    def test_default_uses_pinned_revision(self):
        pinned = tr.pretrained_weights["terramind_v1_tiny"]["hf_hub_revision"]
        assert self._download_revision() == pinned

    def test_user_can_override_revision(self):
        assert self._download_revision(hf_hub_revision="main") == "main"
