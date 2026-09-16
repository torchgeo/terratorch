# LSF Integration Test Runner Documentation

## Overview

`run_lsf_integrationtest.py` is a Python script for submitting Terratorch
integration tests to an LSF cluster. It manages test dependencies, handles job
submission, and provides status tracking for distributed test execution.

## Features

- **Automated test categorization**: Prerequisite, dependent, independent, and
  cleanup tests
- **Dependency management**: Ensures tests run in the correct order
- **Job status tracking**: Monitor test progress and results
- **Flexible configuration**: Support for different Python versions and
  execution environments
- **vLLM test support**: Includes specialized vLLM integration tests

## Usage

### Basic Submission

Test a specific branch from GitHub:

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /path/to/logs \
    --venv-base-dir /path/to/venv
```

Test local code (without cloning from GitHub):

```bash
python3 scripts/run_lsf_integrationtest.py \
    --output-dir /path/to/logs \
    --venv-base-dir /path/to/venv
```

### Required Arguments

- `--output-dir`: Directory for storing test logs (required for submit mode)
- `--venv-base-dir`: Path to virtual environment containing tox (required for
  submit mode)

### Optional Arguments

- `--branch-name`: Git branch name to test. When provided, the specified branch
  is cloned from GitHub and tested. When omitted, tests run against the local
  code in the current directory, allowing you to test uncommitted changes.
- `--python-version`: Python version for tox environments (choices: `py310`,
  `py311`, `py312`, `py313`; default: `py312`)
- `--execution-tag`: Tag for this execution (creates subfolder in output_dir;
  defaults to timestamp `run_YYYYMMDD_HHMMSS`)
- `--test-file`: Path to test file relative to repository root (default:
  `integrationtests/test_base_set.py`)
- `--no-cleanup`: Skip running the cleanup test
- `--cleanup-tox-venv`: Cleanup tox virtual environments after test completion
  (sets `CLEANUP_VENV=1`). This deletes the entire tox environment directory to
  free up disk space.
- `--terratorch-tmp-root`: Path to temporary root directory (sets
  `TERRATORCH_TMP_ROOT` environment variable)
- `--test-name`: Run only a specific test by name (e.g., `test_models_fit`,
  `integration-tests-vllm-release`)
- `--verbose`, `-v`: Enable verbose output showing detailed progress
- `--check-status`: Check status of jobs from a previous run (provide the output
  directory path)

## Test Categories

### 1. Prerequisite Test

- **Test**: `test_models_fit`
- **Purpose**: Creates model checkpoints required by dependent tests
- **Execution**: Runs first with exclusive GPU access

### 2. Dependent Tests

Tests that require checkpoints from `test_models_fit`:

- `test_latest_terratorch_version_buildings_predict`
- `test_latest_terratorch_version_floods_predict`
- `test_latest_terratorch_version_burnscars_predict`

**Execution**: Wait for `test_models_fit` to complete successfully before
starting

### 3. Independent Tests

Tests that can run immediately without dependencies:

- All other tests from `test_base_set.py`
- vLLM test environments:
  - `integration-tests-vllm-release`
  - `vllm-tests-tt-main`

**Execution**: Start immediately in parallel

### 4. Cleanup Test

- **Test**: `test_cleanup`
- **Purpose**: Clean up resources after all dependent tests complete
- **Execution**: Runs last, waits for all dependent tests to finish (regardless
  of success/failure)

## Examples

### Submit All Tests

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name feature-branch \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --python-version py312 \
    --verbose
```

### Submit Single Test

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --test-name test_models_fit
```

### Submit vLLM Test

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --test-name integration-tests-vllm-release
```

### Skip Cleanup Test

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --no-cleanup
```

### Use Custom Temporary Directory

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --terratorch-tmp-root /custom/tmp/path
```

### Cleanup Tox Virtual Environments After Tests

```bash
python3 scripts/run_lsf_integrationtest.py \
    --branch-name main \
    --output-dir /dccstor/terratorch/logs \
    --venv-base-dir /dccstor/terratorch/venv \
    --cleanup-tox-venv
```

This deletes the entire tox environment directory (`.tox/{env_name}`) after each
test completes, freeing up disk space. Useful in distributed environments with
limited disk quotas.

**Note:** Pip and uv package caching is automatically disabled by default to
prevent cache directories from growing too large in distributed test
environments.

### Check Job Status

```bash
python3 scripts/run_lsf_integrationtest.py \
    --check-status /dccstor/terratorch/logs/run_20260306_120000
```

## Output

### Job Submission

The script outputs:

1. Configuration summary
2. Test categorization
3. Job submission progress (if verbose)
4. Summary table of submitted jobs
5. Instructions for monitoring jobs

Example output:

```
================================================================================
LSF Integration Test Runner - Configuration Summary
================================================================================
Branch name: main
Python version: py312
Output directory: /dccstor/terratorch/logs
Execution tag: run_20260306_120000
Test file: integrationtests/test_base_set.py
Skip cleanup: False
Cleanup tox venv: False

Output folder: /dccstor/terratorch/logs/run_20260306_120000
Validated environment and found 18 tests to run

Submitting jobs to LSF...

Submitted Jobs Summary:
----------------------------------------------------------------------------------------------------
Type            Test Name                                          Job ID          Depends On
----------------------------------------------------------------------------------------------------
Prerequisite    test_models_fit                                    12345           None
Dependent       test_latest_terratorch_version_buildings_predict   12346           done(12345)
Independent     test_surya                                         12347           None
...
----------------------------------------------------------------------------------------------------
Total jobs submitted: 18

Logs directory: /dccstor/terratorch/logs/run_20260306_120000
Check status with: python3 scripts/run_lsf_integrationtest.py --check-status /dccstor/terratorch/logs/run_20260306_120000
Monitor jobs with: bjobs -J 'tt_username_*'
================================================================================
```

### Status Check

```
====================================================================================================
Job Status Report - run_20260306_120000
====================================================================================================

Type            Test Name                                          Job ID          Status       Result
-----------------------------------------------------------------------------------------------------------
Prerequisite    test_models_fit                                    12345           DONE         SUCCESS
Dependent       test_latest_terratorch_version_buildings_predict   12346           DONE         SUCCESS
Independent     test_surya                                         12347           RUN          RUNNING
...
-----------------------------------------------------------------------------------------------------------

Summary:
  Total jobs: 18
  Completed: 15 (successful: 14, failed: 1)
  Running/Pending: 3
  Submit failures: 0

Logs directory: /dccstor/terratorch/logs/run_20260306_120000
====================================================================================================
```

## Job Information Storage

The script saves job metadata to `job_ids.json` in the output directory:

```json
{
  "submission_time": "2026-03-06T12:00:00.000000",
  "branch": "main",
  "execution_tag": "run_20260306_120000",
  "jobs": [
    {
      "type": "Prerequisite",
      "test_name": "test_models_fit",
      "job_id": "12345",
      "dependency": "None"
    },
    ...
  ]
}
```

## LSF Job Configuration

Each job is submitted with:

- **GPU**: 1 GPU (exclusive mode for prerequisite and vLLM test)
- **Resources**: 8 CPUs, 32GB memory
- **Logs**: Separate `.log` and `.err` files per test

## Distributed Cluster Considerations

Each test instance gets:

- Dedicated tox working directory (TOX*WORK_DIR):
  `.tox/{branch_name}*{test_name}`

### Disk Space Management

When running tests in distributed environments with limited disk quotas:

1. **Package Caching**: Pip and uv package caching is **automatically disabled**
   to prevent cache directories from growing too large in distributed
   environments. This is always enabled and cannot be changed.

2. **Virtual Environment Cleanup**: Tox creates complete virtual environments
   for each test, which can consume significant disk space. Use
   `--cleanup-tox-venv` to automatically delete these environments after test
   completion.

3. **virtualenv app_data**: `VIRTUALENV_APP_DATA` is **always set** to
   `$TOX_WORK_DIR/virtualenv_app_data`, keeping virtualenv's seed wheel images on
   the same shared filesystem as the checkout. Left at its default it writes to
   the home cache directory, and on a cluster whose home fileset is quota-limited
   every job then fails within seconds of starting:

   ```
   OSError: [Errno 122] Disk quota exceeded
   ```

   This happens during environment seeding, so it looks like a wholesale test
   failure — all jobs exit with code 2 having run no test code at all. If you see
   that, check the home quota (`dd if=/dev/zero of=$HOME/probe bs=1M count=20`)
   before suspecting the code under test. Note that Hugging Face model downloads
   still default to `~/.cache/huggingface`; set `HF_HOME` if home is tight.

### Environment Variables

The following environment variables are set based on command-line options:

- `CLEANUP_VENV=1`: Set when `--cleanup-tox-venv` is used. Triggers deletion of
  tox environment directory after test completion.
- `UV_NO_CACHE=1`: Always set to disable uv package caching in distributed
  environments.
- `PIP_NO_CACHE_DIR=1`: Always set to disable pip package caching in distributed
  environments.
- `VIRTUALENV_APP_DATA`: Always set to `$TOX_WORK_DIR/virtualenv_app_data` to keep
  virtualenv's seed wheel images off the (often quota-limited) home fileset.
- `TEST_BRANCH`: Set to the branch name. When set, tox environments clone from
  GitHub instead of using local code.
- `TERRATORCH_TMP_ROOT`: Set when `--terratorch-tmp-root` is provided. Specifies
  custom temporary directory location.

#### Relocating test assets off `/dccstor`

The integration tests default to the shared `/dccstor` storage. On clusters where
that mount is unavailable, override these variables to point at a local mirror of
the data (all default to their original `/dccstor` locations, so unset behaviour is
unchanged):

- `TERRATORCH_DATA_ROOT`: Storage prefix that replaces `/dccstor` in the fit
  configs' input-dataset paths (e.g. `geofm-datasets`, `geofm-finetuning`).
  Default: `/dccstor`. Set this to the root of a mirror that preserves the
  `/dccstor/...` sub-paths (e.g. mirror `/dccstor/geofm-datasets/...` to
  `$TERRATORCH_DATA_ROOT/geofm-datasets/...`).
- `TERRATORCH_TEST_MODELS_ROOT`: Location of the pre-trained testing models used by
  the legacy predict tests. Default:
  `$TERRATORCH_DATA_ROOT/terratorch/shared/integrationtests/testing_models`.
- `TERRATORCH_TEST_CHECKPOINTS_ROOT`: Location of the legacy checkpoints. Defaults
  to `TERRATORCH_TEST_MODELS_ROOT`.
- `TERRATORCH_TMP_ROOT`: Writable directory for fit outputs/checkpoints. Default:
  `$TERRATORCH_DATA_ROOT/terratorch/tmp`.

Example (mirror rooted at `/proj/.../terratorch/datasets`):

```bash
export TERRATORCH_DATA_ROOT=/proj/partnership-mat/terratorch/datasets
export TERRATORCH_TMP_ROOT=/proj/partnership-mat/terratorch/tmp
pytest integrationtests/test_base_set.py
```

### Direct Tox Usage

You can also run tox directly with these environment variables:

```bash
# With venv cleanup
CLEANUP_VENV=1 tox -e integration-tests-vllm-release-py312

# With disabled caching
UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1 tox -e integration-tests-vllm-release-py312

# With branch selection
TEST_BRANCH=feature-branch tox -e integration-tests-vllm-release-py312

# All options combined
CLEANUP_VENV=1 UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1 TEST_BRANCH=main tox -e integration-tests-vllm-release-py312
```
