#!/bin/bash
# Submit the full integration suite (integrationtests/test_base_set.py) to a Slurm
# cluster. Slurm counterpart of run_lsf_integrationtest.py.
#
# Usage:
#   scripts/run_slurm_integrationtest.sh [REF]
#
#   REF   branch, tag, commit, or pull/<N>/head (default: main)
#
# Environment overrides (defaults are the bluevela settings):
#   TT_BASE         run directory root          (/proj/partnership-mat/terratorch/pr_review)
#   TT_DATA_ROOT    mirror of the /dccstor data (/proj/partnership-mat/terratorch/datasets)
#   TT_PYTHON       python used to build venv   (/gpfs/ess6000-1/optshare/share/miniconda/bin/python3.12)
#   TT_ACCOUNT      Slurm account               (slurm_grp_partnership_mat)
#   TT_GPU_ARGS     extra sbatch args, GPU jobs (--partition=gpu-mid --qos=slurm_grp_partnership_mat_medium --gres=gpu:1)
#   TT_CPU_ARGS     extra sbatch args, CPU jobs (--partition=cpu --qos=cpu)
#   DRY_RUN=1       print the sbatch commands instead of submitting them
#
# Job graph:
#   tt_setup (cpu: venv + pip install)
#     |-- tt_fit_<model>   one GPU job per test_models_fit parameter, in parallel
#     |     `-- tt_test_latest_terratorch_version_*_predict   (after ALL fit jobs ended)
#     |           `-- tt_test_cleanup                         (after the predict jobs ended)
#     `-- every other test, one GPU job each
#
# The clone uses HTTPS because compute nodes have no SSH key.
set -uo pipefail

REF="${1:-main}"
TT_BASE="${TT_BASE:-/proj/partnership-mat/terratorch/pr_review}"
TT_DATA_ROOT="${TT_DATA_ROOT:-/proj/partnership-mat/terratorch/datasets}"
TT_PYTHON="${TT_PYTHON:-/gpfs/ess6000-1/optshare/share/miniconda/bin/python3.12}"
TT_ACCOUNT="${TT_ACCOUNT:-slurm_grp_partnership_mat}"
TT_GPU_ARGS="${TT_GPU_ARGS:---partition=gpu-mid --qos=slurm_grp_partnership_mat_medium --gres=gpu:1}"
TT_CPU_ARGS="${TT_CPU_ARGS:---partition=cpu --qos=cpu}"

TEST_FILE=integrationtests/test_base_set.py
# These tests need the checkpoints that test_models_fit writes.
DEPENDENT="test_latest_terratorch_version_buildings_predict test_latest_terratorch_version_floods_predict test_latest_terratorch_version_burnscars_predict"

RUN_DIR="$TT_BASE/full_$(date +%Y%m%d_%H%M%S)"
SRC="$RUN_DIR/src"
VENV="$RUN_DIR/venv"
LOGS="$RUN_DIR/logs"
mkdir -p "$RUN_DIR/tmp" "$RUN_DIR/hf" "$LOGS"

# Submit a job and print its id. With DRY_RUN=1 nothing is submitted.
sb() {
    if [ "${DRY_RUN:-0}" = 1 ]; then
        echo "[dry-run] sbatch $*" >&2
        echo "DRY$RANDOM"
    else
        sbatch --parsable "$@"
    fi
}

git clone -q https://github.com/torchgeo/terratorch.git "$SRC" || { echo "CLONE_FAIL" >&2; exit 1; }
cd "$SRC" || exit 1
git fetch -q origin "$REF" && git checkout -q -f FETCH_HEAD || { echo "CHECKOUT_FAIL: $REF" >&2; exit 1; }
git log --oneline -1 | tee "$RUN_DIR/commit.txt"

# Parameters of test_models_fit, read from the file so this never drifts from the tests.
FIT_MODELS=$(python3 - "$TEST_FILE" <<'PY'
import ast
import sys

for node in ast.parse(open(sys.argv[1]).read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "test_models_fit":
        for dec in node.decorator_list:
            if getattr(getattr(dec, "func", None), "attr", "") == "parametrize":
                print(" ".join(ast.literal_eval(dec.args[1])))
PY
)
[ -n "$FIT_MODELS" ] || { echo "Could not find test_models_fit parameters in $TEST_FILE" >&2; exit 1; }
ALL_TESTS=$(grep -E '^def test_' "$TEST_FILE" | sed 's/def //' | cut -d'(' -f1)

ENV_SETUP="export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1 TERRATORCH_DATA_ROOT=$TT_DATA_ROOT TERRATORCH_TMP_ROOT=$RUN_DIR/tmp HF_HOME=$RUN_DIR/hf; source $VENV/bin/activate; cd $SRC"
# --kill-on-invalid-dep: a failed setup cancels everything that depends on it instead of leaving it pending forever.
COMMON="--account=$TT_ACCOUNT --kill-on-invalid-dep=yes"
GPU="$COMMON $TT_GPU_ARGS --cpus-per-task=8 --mem=64G --time=12:00:00"
CPU="$COMMON $TT_CPU_ARGS --cpus-per-task=8 --mem=32G --time=02:00:00"

# shellcheck disable=SC2086  # $CPU/$GPU are intentionally word-split into sbatch options
SETUP=$(sb $CPU -J tt_setup -o "$LOGS/setup.log" --wrap="set -e; export UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1; $TT_PYTHON -m venv $VENV; source $VENV/bin/activate; pip install -q --upgrade pip setuptools wheel; cd $SRC; pip install -q -e '.[test]'; python -c 'import terratorch, torch; print(\"IMPORT_OK\", torch.__version__, torch.version.cuda)'")
echo "setup: $SETUP"

FIT_IDS=""
for M in $FIT_MODELS; do
    # shellcheck disable=SC2086
    J=$(sb $GPU -J "tt_fit_$M" -d "afterok:$SETUP" -o "$LOGS/fit_$M.log" --wrap="$ENV_SETUP; nvidia-smi -L; pytest -v -s '$TEST_FILE::test_models_fit[$M]'")
    FIT_IDS="$FIT_IDS:$J"
done
FIT_IDS=${FIT_IDS#:}
echo "fit jobs: $FIT_IDS"

DEP_IDS=""
for T in $DEPENDENT; do
    grep -q "^def $T(" "$TEST_FILE" || continue
    # shellcheck disable=SC2086
    J=$(sb $GPU -J "tt_$T" -d "afterany:$FIT_IDS" -o "$LOGS/$T.log" --wrap="$ENV_SETUP; pytest -v -s $TEST_FILE -k $T")
    DEP_IDS="$DEP_IDS:$J"
done
DEP_IDS=${DEP_IDS#:}

for T in $ALL_TESTS; do
    case " test_models_fit test_cleanup $DEPENDENT " in *" $T "*) continue ;; esac
    # shellcheck disable=SC2086
    sb $GPU -J "tt_$T" -d "afterok:$SETUP" -o "$LOGS/$T.log" --wrap="$ENV_SETUP; pytest -v -s $TEST_FILE -k $T" > /dev/null
done

if echo "$ALL_TESTS" | grep -qx test_cleanup && [ -n "$DEP_IDS" ]; then
    # shellcheck disable=SC2086
    sb $GPU -J tt_test_cleanup -d "afterany:$DEP_IDS" -o "$LOGS/test_cleanup.log" --wrap="$ENV_SETUP; pytest -v -s $TEST_FILE -k test_cleanup" > /dev/null
fi

echo "RUN_DIR=$RUN_DIR"
echo "Monitor:  squeue -u \$USER -o '%.10i %.40j %.8T %.10M %R'"
echo "Results:  grep -E 'passed|failed|error' $LOGS/*.log"
