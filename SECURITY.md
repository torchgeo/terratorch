# Security Policy

## Supported Versions

TerraTorch is developed on `main` and released from there. Security fixes are
backported to the latest minor release line only.

| Version | Supported |
| ------- | --------- |
| 1.2.x   | ✅        |
| < 1.2   | ❌        |

## Reporting a Vulnerability

**Please do not open a public issue for security problems.**

Report vulnerabilities privately through GitHub's private vulnerability
reporting:

1. Go to the [Security tab](https://github.com/torchgeo/terratorch/security)
   of this repository.
2. Click **Report a vulnerability**.
3. Describe the issue, the affected version, and a minimal reproduction if
   possible.

You can expect:

- an acknowledgement within **5 business days**,
- an assessment and a remediation plan within **30 days**,
- credit in the published advisory unless you prefer to stay anonymous.

If you cannot use GitHub private reporting, contact the maintainers listed in
[`pyproject.toml`](pyproject.toml) directly.

## Scope

TerraTorch loads model checkpoints and configuration files, and downloads
pretrained weights from external hubs. Reports in these areas are especially
welcome:

- unsafe deserialization (`torch.load`, `pickle`, `joblib`, `yaml.load`),
- path traversal or arbitrary file write during checkpoint/config handling,
- code execution through model or tokenizer configuration,
- supply-chain issues in the weight download path.

## Handling of Model Checkpoints

TerraTorch treats checkpoints as **untrusted input**:

- `torch.load()` is always called with `weights_only=True`.
- Only load checkpoints from sources you trust. A checkpoint is executable
  content in disguise; `weights_only=True` reduces but does not eliminate the
  risk for older PyTorch versions.

## Dependency and Supply-Chain Policy

- Pretrained weights are downloaded with an explicit `revision` (commit SHA)
  wherever the upstream repository is public, so a moved tag cannot silently
  change the weights. Repositories that require authentication are downloaded
  from their default branch and should be pinned once a token is available.
- Dependabot monitors `pip` and `github-actions` dependencies weekly.
- CodeQL runs on every push to `main` and on every pull request.
- Pull requests that change dependencies are checked by the dependency review
  action; new dependencies with known vulnerabilities or incompatible licenses
  block the merge.
