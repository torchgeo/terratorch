#!/usr/bin/env python3
"""Check that every torch.load() and get_state_dict() call passes weights_only=True """

import ast
import subprocess
import sys

bad = []
for path in subprocess.check_output(["git", "ls-files", "*.py"], text=True).split():
    for node in ast.walk(ast.parse(open(path).read(), path)):
        if (
            isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                    and node.func.attr == "load"
                )
                or (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get_state_dict"
                )
            )
            and not any(
                kw.arg == "weights_only"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
        ):
            bad.append(f"{path}:{node.lineno}")

if bad:
    print("torch.load() or get_state_dict() without weights_only=True:\n" + "\n".join(bad))
    sys.exit(1)
