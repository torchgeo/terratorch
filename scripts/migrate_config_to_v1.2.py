#!/usr/bin/env python3
"""
TerraTorch YAML migration script to TerraTorch v1.2

Usage:
  python migrate_terratorch_yaml.py input.yaml > output.yaml
  python migrate_terratorch_yaml.py input.yaml -i   # in-place
"""

import argparse
import sys
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True


REPLACEMENTS = {
    "img_grep": "image_grep",
    "False": "false",
    "ReduceLROnPlateau": "lightning.pytorch.cli.ReduceLROnPlateau",
    "ToTensorV2": "albumentations.pytorch.transforms.ToTensorV2",
    "TensorBoardLogger": "lightning.pytorch.loggers.TensorBoardLogger",
    "LearningRateMonitor": "lightning.pytorch.callbacks.LearningRateMonitor",
    "EarlyStopping": "lightning.pytorch.callbacks.EarlyStopping",
}


REMOVE_CALLBACKS = {
    "RichProgressBar",
    "lightning.pytorch.callbacks.RichProgressBar",
}


def deep_replace(obj):
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            # drop RichProgressBar entries
            if k == "class_path" and v in REMOVE_CALLBACKS:
                return None
            new_k = deep_replace(k)
            new_v = deep_replace(v)
            if new_v is not None:
                new[new_k] = new_v
        return new

    if isinstance(obj, list):
        out = []
        for item in obj:
            replaced = deep_replace(item)
            if replaced is not None:
                out.append(replaced)
        return out

    if isinstance(obj, str):
        for src, dst in REPLACEMENTS.items():
            obj = obj.replace(src, dst)
        return obj

    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file")
    parser.add_argument("-i", "--in-place", action="store_true")
    args = parser.parse_args()

    with open(args.yaml_file) as f:
        data = yaml.load(f)

    migrated = deep_replace(data)

    if args.in_place:
        with open(args.yaml_file, "w") as f:
            yaml.dump(migrated, f)
    else:
        yaml.dump(migrated, sys.stdout)


if __name__ == "__main__":
    main()
