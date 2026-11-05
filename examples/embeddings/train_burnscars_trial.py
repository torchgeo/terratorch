#!/usr/bin/env python3
"""
Trial script for iterate HPO over the burn scars downstream segmentation task.

iterate passes sampled and static parameters as environment variables:
  ITERATE_PARAM_LR       – learning rate sampled by Optuna
  ITERATE_PARAM_CONFIG   – base terratorch config file (static)
  ITERATE_PARAM_EPOCHS   – number of training epochs per trial (static)
  ITERATE_TRIAL_NUMBER   – integer trial index
  ITERATE_OUT_FILE       – path where metrics must be written (name: value)

The script:
  1. Reads parameters from ITERATE_PARAM_* environment variables.
  2. Invokes the terratorch CLI via subprocess, overriding the learning rate
     with a jsonargparse dotted-key flag:
       terratorch fit -c <config> \\
           --optimizer.init_args.lr <lr> \\
           --trainer.max_epochs <epochs>
  3. Parses Lightning's CSV log to extract the best validation loss.
  4. Writes the metric to ITERATE_OUT_FILE so iterate can read it.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

# Script directory – used to resolve config paths regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent


def get_params():
    """Read trial parameters from ITERATE_PARAM_* environment variables."""
    lr_str = os.environ.get("ITERATE_PARAM_LR")
    if lr_str is None:
        print("ERROR: ITERATE_PARAM_LR not set", file=sys.stderr)
        sys.exit(1)
    lr = float(lr_str)

    config = os.environ.get(
        "ITERATE_PARAM_CONFIG",
        "downstream_segmentation_burnscars.yaml",
    )
    # Resolve config path: prefer ITERATE_NB_DIR (set by the notebook) so the
    # config file is found in the notebook's directory even when the trial
    # script runs from a different location (e.g. /home/sagemaker-user/).
    config_path = Path(config)
    if not config_path.is_absolute():
        base = Path(os.environ.get("ITERATE_NB_DIR", str(SCRIPT_DIR)))
        config_path = base / config_path

    epochs = int(os.environ.get("ITERATE_PARAM_EPOCHS", "5"))
    return lr, config_path, epochs


def find_metrics_csv(root: Path):
    """Return the most recently modified metrics.csv under *root*."""
    candidates = sorted(
        root.rglob("metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def best_val_loss(metrics_csv: Path) -> float:
    """Parse Lightning's metrics.csv and return the minimum val/loss."""
    values = []
    with metrics_csv.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = row.get("val/loss", "").strip()
            if cell:
                try:
                    values.append(float(cell))
                except ValueError:
                    pass
    if not values:
        raise RuntimeError(f"No 'val/loss' rows found in {metrics_csv}")
    return min(values)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Read parameters from ITERATE_PARAM_* environment variables
    # ------------------------------------------------------------------ #
    lr, config_path, epochs = get_params()

    trial_num = os.environ.get("ITERATE_TRIAL_NUMBER", "?")
    out_file  = os.environ.get("ITERATE_OUT_FILE")

    print(f"[trial {trial_num}] lr={lr:.2e}  epochs={epochs}  config={config_path}")

    # ------------------------------------------------------------------ #
    # 2. Run terratorch fit with jsonargparse override for learning rate.
    #    The dotted key `optimizer.init_args.lr` follows Lightning-CLI /
    #    jsonargparse conventions and works with any optimizer block.
    # ------------------------------------------------------------------ #
    log_dir = SCRIPT_DIR / f"hpo_trial_{trial_num}"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Write metrics.csv to a fully known path by passing an explicit CSVLogger.
    # Lightning's default root_dir handling can vary; this is deterministic.
    # CSVLogger(save_dir=log_dir, name="", version=0) writes to:
    #   {save_dir}/version_0/metrics.csv
    metrics_csv_path = log_dir / "version_0" / "metrics.csv"

    logger_json = (
        '{"class_path":"lightning.pytorch.loggers.CSVLogger",'
        '"init_args":{"save_dir":"' + str(log_dir) + '",'
        '"name":"","version":0}}'
    )

    cmd = [
        sys.executable, "-m", "terratorch",
        "fit",
        "-c", str(config_path),
        "--optimizer.init_args.lr", str(lr),
        "--trainer.max_epochs", str(epochs),
        "--trainer.default_root_dir", str(log_dir),
        "--trainer.logger", logger_json,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"[trial {trial_num}] terratorch failed (exit {result.returncode})",
              file=sys.stderr)
        sys.exit(result.returncode)

    # ------------------------------------------------------------------ #
    # 3. Extract best validation loss from Lightning's auto-generated CSV
    # ------------------------------------------------------------------ #
    # Prefer the explicit logger path; fall back to rglob search.
    metrics_csv = metrics_csv_path if metrics_csv_path.exists() else find_metrics_csv(log_dir)
    if metrics_csv is None:
        print(f"[trial {trial_num}] ERROR: no metrics.csv found under {log_dir}",
              file=sys.stderr)
        sys.exit(1)

    val_loss = best_val_loss(metrics_csv)
    print(f"[trial {trial_num}] best val_loss = {val_loss:.6f}")

    # ------------------------------------------------------------------ #
    # 4. Write metric to ITERATE_OUT_FILE (iterate reads this for Optuna)
    # ------------------------------------------------------------------ #
    if out_file:
        with open(out_file, "w") as fh:
            fh.write(f"val_loss: {val_loss}\n")
        print(f"[trial {trial_num}] metrics written to {out_file}")
    else:
        # Fallback: print in the expected format if env var is not set
        print(f"val_loss: {val_loss}")


if __name__ == "__main__":
    main()
