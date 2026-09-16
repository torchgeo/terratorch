#!/usr/bin/env python3
"""
LSF Integration Test Runner for Terratorch

This script manages the submission of integration tests to LSF with proper dependencies.
It handles test_models_fit as a prerequisite for dependent tests and manages cleanup.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Constants for test categorization
DEPENDENT_TESTS = [
    "test_latest_terratorch_version_buildings_predict",
    "test_latest_terratorch_version_floods_predict",
    "test_latest_terratorch_version_burnscars_predict",
]

VLLM_TESTS = [
    "integration-tests-vllm-release",
    "integration-tests-vllm-main",
    "vllm-tests-tt-main",
]

PREREQUISITE_TEST = "test_models_fit"
CLEANUP_TEST = "test_cleanup"


def extract_test_names(test_file_path):
    """Extract test function names from the test file."""
    test_file = Path(test_file_path)

    if not test_file.exists():
        print(f"Error: Test file not found at {test_file}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)

    test_names = []
    with test_file.open("r") as f:
        for line in f:
            match = re.match(r"^def (test_\w+)", line.strip())
            if match:
                test_names.append(match.group(1))

    return test_names


def submit_lsf_job(
    job_name,
    log_file,
    err_file,
    command,
    dependency=None,
    gpu_config="num=1:gmem=16G",
    terminate_on_dependency_failure=False,
):
    """Submit a job to LSF and return the job ID.

    Default gpu_config requires at least 16GB of GPU memory.
    """
    bsub_cmd = [
        "bsub",
        "-gpu",
        gpu_config,
        "-R",
        "rusage[cpu=8, mem=32GB]",
        "-J",
        job_name,
        "-o",
        str(log_file),
        "-e",
        str(err_file),
    ]

    if dependency:
        bsub_cmd.extend(["-w", dependency])
        # Add -ti flag to terminate job if dependency fails
        if terminate_on_dependency_failure:
            bsub_cmd.append("-ti")

    bsub_cmd.append(command)

    result = subprocess.run(bsub_cmd, capture_output=True, text=True)

    # Extract job ID from output like "Job <12345> is submitted..."
    match = re.search(r"Job <(\d+)>", result.stdout)
    if match:
        return match.group(1)

    print(f"Warning: Could not extract job ID from: {result.stdout}")
    return None


def check_job_status(output_dir):
    """Check the status of jobs from a previous run."""
    output_path = Path(output_dir).resolve()
    jobs_file = output_path / "job_ids.json"

    if not jobs_file.exists():
        print(f"Error: Job IDs file not found at {jobs_file}")
        print("Make sure you're pointing to the correct output directory.")
        sys.exit(1)

    # Load job information
    with jobs_file.open("r") as f:
        job_data = json.load(f)

    print("=" * 100)
    print(f"Job Status Report - {output_path.name}")
    print("=" * 100)
    print()

    # Get status for all jobs
    job_ids = [job["job_id"] for job in job_data["jobs"] if job["job_id"] != "FAILED"]

    if not job_ids:
        print("No valid job IDs found.")
        return

    # Query bjobs for all job IDs at once
    try:
        result = subprocess.run(
            ["bjobs", "-a", "-o", "jobid stat exit_code delimiter='|'"] + job_ids,
            capture_output=True,
            text=True,
            check=False,
        )

        # Parse bjobs output
        job_status_map = {}
        lines = result.stdout.strip().split("\n")

        # Debug: print raw output
        if os.environ.get("DEBUG_BJOBS"):
            print("\nDEBUG: bjobs output:")
            print(result.stdout)
            print()

        for line in lines[1:]:  # Skip header
            if "|" in line:
                parts = line.split("|")
                if len(parts) >= 2:
                    job_id = parts[0].strip()
                    status = parts[1].strip()
                    # Exit code might be empty, "-", or a number
                    exit_code = parts[2].strip() if len(parts) >= 3 else ""
                    # Normalize empty or "-" to "0" for DONE jobs
                    if status == "DONE" and (not exit_code or exit_code == "-"):
                        exit_code = "0"
                    job_status_map[job_id] = (status, exit_code)
    except Exception as e:
        print(f"Error querying job status: {e}")
        job_status_map = {}

    # Print status table
    print(f"{'Type':<15} {'Test Name':<50} {'Job ID':<15} {'Status':<12} {'Result':<15}")
    print("-" * 107)

    for job in job_data["jobs"]:
        job_type = job["type"]
        test_name = job["test_name"]
        job_id = job["job_id"]

        if job_id == "FAILED":
            status = "N/A"
            result_str = "SUBMIT FAILED"
        elif job_id in job_status_map:
            status, exit_code = job_status_map[job_id]

            if status == "DONE":
                if exit_code == "0":
                    result_str = "SUCCESS"
                else:
                    result_str = f"FAILED (exit {exit_code})"
            elif status == "EXIT":
                result_str = f"FAILED (exit {exit_code})"
            elif status in ["PEND", "RUN"]:
                result_str = "RUNNING"
            else:
                result_str = status
        else:
            status = "UNKNOWN"
            result_str = "NOT FOUND"

        print(f"{job_type:<15} {test_name:<50} {job_id:<15} {status:<12} {result_str:<15}")

    print("-" * 107)
    print()

    # Summary statistics
    total_jobs = len(job_data["jobs"])
    failed_submits = sum(1 for job in job_data["jobs"] if job["job_id"] == "FAILED")

    completed = sum(
        1 for job in job_data["jobs"] if job["job_id"] in job_status_map and job_status_map[job["job_id"]][0] == "DONE"
    )
    successful = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map
        and job_status_map[job["job_id"]][0] == "DONE"
        and job_status_map[job["job_id"]][1] == "0"
    )
    failed = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map
        and (job_status_map[job["job_id"]][0] in ["DONE", "EXIT"])
        and job_status_map[job["job_id"]][1] != "0"
    )
    running = sum(
        1
        for job in job_data["jobs"]
        if job["job_id"] in job_status_map and job_status_map[job["job_id"]][0] in ["PEND", "RUN"]
    )

    print("Summary:")
    print(f"  Total jobs: {total_jobs}")
    print(f"  Completed: {completed} (successful: {successful}, failed: {failed})")
    print(f"  Running/Pending: {running}")
    print(f"  Submit failures: {failed_submits}")
    print()
    print(f"Logs directory: {output_path}")
    print("=" * 100)


def build_cache_exports(tox_work_dir, disable_pip_cache):
    """Build the export string for the caches tox and virtualenv write to.

    virtualenv keeps its seed wheel images in an app_data directory that defaults
    to the user's home cache. On clusters where the home fileset is quota-limited
    that quota is usually the first thing to run out, and every job then dies with
    "OSError: [Errno 122] Disk quota exceeded" while seeding the environment,
    before a single test runs. Pinning app_data beside the tox work directory
    keeps it on the same (large) shared filesystem as the checkout.
    """
    exports = f" && export VIRTUALENV_APP_DATA={tox_work_dir}/virtualenv_app_data"
    if disable_pip_cache:
        exports += " && export UV_NO_CACHE=1 && export PIP_NO_CACHE_DIR=1"
    return exports


def build_test_command(
    activate_cmd,
    branch_name,
    test_name,
    tox_work_dir,
    env_exports,
    python_version,
    cleanup_venv=False,
    disable_pip_cache=False,
):
    """Build the command string for running a test."""
    # Only export TEST_BRANCH if it's set (allows testing local code when not set)
    branch_export = f" && export TEST_BRANCH={branch_name}" if branch_name else ""
    cleanup_export = " && export CLEANUP_VENV=1" if cleanup_venv else ""
    cache_exports = build_cache_exports(tox_work_dir, disable_pip_cache)
    return f"/bin/bash -c 'set -e; {activate_cmd}{branch_export} && export TEST_FUNCTION={test_name} && export TOX_WORK_DIR={tox_work_dir}{env_exports}{cleanup_export}{cache_exports} && tox -r -e integration-tests-base-set-{python_version}; exit $?'"


def build_vllm_command(
    activate_cmd, branch_name, tox_work_dir, tox_env, python_version, cleanup_venv=False, disable_pip_cache=False
):
    """Build the command string for running a vLLM test."""
    # Only export TEST_BRANCH if it's set (allows testing local code when not set)
    exports = f"export TOX_WORK_DIR={tox_work_dir}"
    if branch_name:
        exports = f"export TEST_BRANCH={branch_name} && {exports}"
    if cleanup_venv:
        exports += " && export CLEANUP_VENV=1"
    exports += build_cache_exports(tox_work_dir, disable_pip_cache)
    return f"/bin/bash -c 'set -e; {activate_cmd} && {exports} && tox -r -e {tox_env}-{python_version}; exit $?'"


def build_job_names(user, branch_name, test_name):
    """Build job name and tox work directory for a test."""
    job_name = f"tt_{user}_{test_name}"
    # Use "local" as prefix when no branch name is provided
    branch_prefix = branch_name if branch_name else "local"
    tox_work_dir = f".tox/{branch_prefix}_{test_name}"
    return job_name, tox_work_dir


def build_log_paths(log_dir, test_name):
    """Build log and error file paths for a test."""
    return log_dir / f"{test_name}.log", log_dir / f"{test_name}.err"


def record_job_submission(submitted_jobs, job_type, test_name, job_id, dependency, verbose):
    """Record a job submission and print status if verbose."""
    if job_id:
        submitted_jobs.append((job_type, test_name, job_id, dependency))
        if verbose:
            print(f"      Job submitted with ID: {job_id}")
        return True
    else:
        submitted_jobs.append((job_type, test_name, "FAILED", dependency))
        if verbose:
            print("      Failed to submit job")
        return False


def submit_and_record_job(job_kwargs, submitted_jobs, job_type, test_name, dependency, verbose, dependent_job_ids=None):
    """Submit a job to LSF and record the result."""
    job_id = submit_lsf_job(**job_kwargs)
    success = record_job_submission(submitted_jobs, job_type, test_name, job_id, dependency, verbose)
    if success and dependent_job_ids is not None:
        dependent_job_ids.append(job_id)
    return job_id


def submit_dependent_tests(
    dependent_test_list,
    models_fit_test,
    models_fit_job_id,
    single_test_mode,
    verbose,
    user,
    activate_cmd,
    branch_name,
    env_exports,
    python_version,
    log_dir,
    submitted_jobs,
    dependent_job_ids,
    cleanup_venv=False,
    disable_pip_cache=False,
):
    """Submit dependent tests that normally wait for test_models_fit."""
    if not dependent_test_list or not models_fit_job_id:
        return

    if verbose:
        status_msg = "independently" if single_test_mode else f"(will wait for {models_fit_test})"
        print(f"\n[2/4] Submitting {len(dependent_test_list)} dependent test(s) {status_msg}:")

    for idx, test_name in enumerate(dependent_test_list, 1):
        verbose_msg = f"  [{idx}/{len(dependent_test_list)}] Submitting: {test_name}" if verbose else None

        # Add dependency only if not in single test mode
        dependency = f"done({models_fit_job_id})" if not single_test_mode else None
        terminate_on_failure = True if not single_test_mode else False

        submit_single_test(
            test_name=test_name,
            job_type="Dependent",
            verbose=verbose,
            user=user,
            activate_cmd=activate_cmd,
            branch_name=branch_name,
            env_exports=env_exports,
            python_version=python_version,
            log_dir=log_dir,
            submitted_jobs=submitted_jobs,
            dependent_job_ids=dependent_job_ids,
            verbose_message=verbose_msg,
            dependency=dependency,
            terminate_on_dependency_failure=terminate_on_failure,
            cleanup_venv=cleanup_venv,
            disable_pip_cache=disable_pip_cache,
        )


def submit_single_test(
    test_name,
    job_type,
    verbose,
    user,
    activate_cmd,
    branch_name,
    env_exports,
    python_version,
    log_dir,
    submitted_jobs,
    dependent_job_ids=None,
    gpu_config=None,
    verbose_message=None,
    dependency=None,
    terminate_on_dependency_failure=False,
    command_builder=None,
    cleanup_venv=False,
    disable_pip_cache=False,
):
    """Submit a single test job (prerequisite, independent, dependent, cleanup, etc.).

    Args:
        test_name: Name of the test to submit
        job_type: Type of job (e.g., "Prerequisite", "Dependent", "Independent", "Cleanup")
        verbose: Whether to print verbose output
        user: Username for job naming
        activate_cmd: Command to activate virtual environment
        branch_name: Git branch name
        env_exports: Environment variable exports string
        python_version: Python version for tox
        log_dir: Directory for log files
        submitted_jobs: List to track submitted jobs
        dependent_job_ids: Optional list to append job ID to (for tracking dependencies)
        gpu_config: Optional GPU configuration string
        verbose_message: Optional message to print if verbose
        dependency: Optional LSF dependency string (e.g., "done(12345)")
        terminate_on_dependency_failure: Whether to terminate if dependency fails
        command_builder: Optional custom command builder function (defaults to build_test_command)

    Returns:
        Job ID of the submitted job
    """
    if verbose and verbose_message:
        print(verbose_message)

    job_name, tox_work_dir = build_job_names(user, branch_name, test_name)
    log_file, err_file = build_log_paths(log_dir, test_name)

    # Use custom command builder if provided, otherwise use default
    if command_builder is not None:
        command = command_builder(
            activate_cmd, tox_work_dir, test_name, python_version, cleanup_venv, disable_pip_cache
        )
    else:
        command = build_test_command(
            activate_cmd,
            branch_name,
            test_name,
            tox_work_dir,
            env_exports,
            python_version,
            cleanup_venv,
            disable_pip_cache,
        )

    job_kwargs = {
        "job_name": job_name,
        "log_file": log_file,
        "err_file": err_file,
        "command": command,
    }

    # Only add gpu_config if provided
    if gpu_config is not None:
        job_kwargs["gpu_config"] = gpu_config

    # Add dependency if provided
    if dependency is not None:
        job_kwargs["dependency"] = dependency
        job_kwargs["terminate_on_dependency_failure"] = terminate_on_dependency_failure

    # Determine dependency info for recording
    dependency_info = dependency if dependency else "None"

    job_id = submit_and_record_job(
        job_kwargs, submitted_jobs, job_type, test_name, dependency_info, verbose, dependent_job_ids
    )
    return job_id


def submit_cleanup_test(
    cleanup_test,
    dependent_job_ids,
    skip_cleanup,
    verbose,
    user,
    activate_cmd,
    branch_name,
    env_exports,
    python_version,
    log_dir,
    submitted_jobs,
    cleanup_venv=False,
    disable_pip_cache=False,
):
    """Submit the cleanup test that runs after all dependent tests."""
    if not cleanup_test or not dependent_job_ids or skip_cleanup:
        if skip_cleanup and verbose:
            print("\n[3/4] Skipping cleanup test (--no-cleanup flag set)")
        return

    verbose_msg = None
    if verbose:
        verbose_msg = f"\n[3/4] Submitting cleanup test: {cleanup_test}\n  Will wait for {len(dependent_job_ids)} job(s) to complete"

    # Build dependency condition - wait for all dependent jobs to end
    cleanup_dependency = " && ".join([f"ended({jid})" for jid in dependent_job_ids])

    submit_single_test(
        test_name=cleanup_test,
        job_type="Cleanup",
        verbose=verbose,
        user=user,
        activate_cmd=activate_cmd,
        branch_name=branch_name,
        env_exports=env_exports,
        python_version=python_version,
        log_dir=log_dir,
        submitted_jobs=submitted_jobs,
        verbose_message=verbose_msg,
        dependency=cleanup_dependency,
        cleanup_venv=cleanup_venv,
        disable_pip_cache=disable_pip_cache,
    )


def submit_independent_tests(
    independent_test_list,
    vllm_tests_to_run,
    verbose,
    user,
    activate_cmd,
    branch_name,
    terratorch_tmp_root,
    python_version,
    log_dir,
    submitted_jobs,
    cleanup_venv=False,
    disable_pip_cache=False,
):
    """Submit independent tests from test file and vLLM test environments."""
    # Build environment variable export string for TERRATORCH_TMP_ROOT
    env_exports = ""
    if terratorch_tmp_root:
        env_exports = f" && export TERRATORCH_TMP_ROOT={terratorch_tmp_root}"

    total_independent = len(independent_test_list) + len(vllm_tests_to_run)
    if not (independent_test_list or vllm_tests_to_run):
        return

    if verbose:
        print(f"\n[4/4] Submitting {total_independent} independent test(s) (run immediately):")

    # Submit tests from test file
    for idx, test_name in enumerate(independent_test_list, 1):
        verbose_msg = f"  [{idx}/{total_independent}] Submitting: {test_name}" if verbose else None

        submit_single_test(
            test_name=test_name,
            job_type="Independent",
            verbose=verbose,
            user=user,
            activate_cmd=activate_cmd,
            branch_name=branch_name,
            env_exports=env_exports,
            python_version=python_version,
            log_dir=log_dir,
            submitted_jobs=submitted_jobs,
            verbose_message=verbose_msg,
            cleanup_venv=cleanup_venv,
            disable_pip_cache=disable_pip_cache,
        )

    # Submit vLLM test environments
    for idx, tox_env in enumerate(vllm_tests_to_run, len(independent_test_list) + 1):
        verbose_msg = f"  [{idx}/{total_independent}] Submitting: {tox_env}" if verbose else None

        # Create a custom command builder for vLLM tests
        def vllm_command_builder(activate_cmd, tox_work_dir, tox_env, python_version, cleanup_venv, disable_pip_cache):
            return build_vllm_command(
                activate_cmd, branch_name, tox_work_dir, tox_env, python_version, cleanup_venv, disable_pip_cache
            )

        submit_single_test(
            test_name=tox_env,
            job_type="Independent",
            verbose=verbose,
            user=user,
            activate_cmd=activate_cmd,
            branch_name=branch_name,
            env_exports=env_exports,
            python_version=python_version,
            log_dir=log_dir,
            submitted_jobs=submitted_jobs,
            gpu_config="num=1:mode=exclusive_process",
            verbose_message=verbose_msg,
            command_builder=vllm_command_builder,
            cleanup_venv=cleanup_venv,
            disable_pip_cache=disable_pip_cache,
        )


def main():
    # Determine script location and repository root
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent  # scripts/run_lsf_integrationtest.py -> repo root
    default_test_file = repo_root / "integrationtests" / "test_base_set.py"

    parser = argparse.ArgumentParser(
        description="Submit Terratorch integration tests to LSF or check status of previous runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--branch-name",
        help="Git branch name to test. When provided, clones and tests the specified branch from GitHub. When omitted, tests local code in current directory.",
    )
    parser.add_argument(
        "--python-version",
        default="py312",
        choices=["py310", "py311", "py312", "py313"],
        help="Python version for tox environments (default: py312)",
    )
    parser.add_argument("--output-dir", help="Output directory for storing test logs (required for submit mode)")
    parser.add_argument(
        "--execution-tag",
        help="Tag for this execution (creates subfolder in output_dir; defaults to timestamp run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--venv-base-dir", help="Path to virtual environment containing tox (required for submit mode)")
    parser.add_argument(
        "--test-file",
        default=str(default_test_file.relative_to(repo_root)),
        help=f"Path to the test file relative to repository root (default: {default_test_file.relative_to(repo_root)})",
    )
    parser.add_argument("--no-cleanup", action="store_true", help="Skip running the cleanup test")
    parser.add_argument(
        "--terratorch-tmp-root",
        metavar="PATH",
        help="Path to temporary root directory (sets TERRATORCH_TMP_ROOT environment variable for non-vLLM tests)",
    )
    parser.add_argument(
        "--check-status",
        metavar="OUTPUT_DIR",
        help="Check status of jobs from a previous run (provide the output directory path)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing detailed progress for each step",
    )
    parser.add_argument(
        "--test-name",
        metavar="TEST_NAME",
        help="Run only a specific test by name (e.g., test_models_fit, integration-tests-vllm-release). When specified, only this test will be executed.",
    )
    parser.add_argument(
        "--cleanup-tox-venv",
        action="store_true",
        help="Cleanup tox virtual environments after test completion (sets CLEANUP_VENV=1)",
    )

    args = parser.parse_args()

    # Store verbose flag for use throughout
    verbose = args.verbose

    # If checking status, do that and exit
    if args.check_status:
        check_job_status(args.check_status)
        return

    # Validate mandatory arguments with graceful error messages
    missing_args = []
    if not args.output_dir:
        missing_args.append("--output-dir")
    if not args.venv_base_dir:
        missing_args.append("--venv-base-dir")

    if missing_args:
        print("=" * 80)
        print("ERROR: Missing required arguments")
        print("=" * 80)
        print(f"The following required arguments are missing: {', '.join(missing_args)}")
        print()
        print("Usage example:")
        print(f"  python3 {sys.argv[0]} --branch-name main --output-dir /path/to/logs --venv-base-dir /path/to/venv")
        print()
        print("For full help, run:")
        print(f"  python3 {sys.argv[0]} --help")
        print("=" * 80)
        sys.exit(1)

    branch_name = args.branch_name
    python_version = args.python_version
    skip_cleanup = args.no_cleanup
    cleanup_venv = args.cleanup_tox_venv
    disable_pip_cache = True  # Always disable pip cache in distributed environments

    # Handle TERRATORCH_TMP_ROOT
    terratorch_tmp_root = args.terratorch_tmp_root

    # Generate execution tag if not provided
    if args.execution_tag:
        execution_tag = args.execution_tag
    else:
        execution_tag = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 80)
    print("LSF Integration Test Runner - Configuration Summary")
    print("=" * 80)
    print(f"Branch name: {branch_name if branch_name else 'local code (not set)'}")
    print(f"Python version: {python_version}")
    print(f"Output directory: {args.output_dir}")
    print(f"Execution tag: {execution_tag}")
    print(f"Test file: {args.test_file}")
    print(f"Skip cleanup: {skip_cleanup}")
    print(f"Cleanup tox venv: {cleanup_venv}")
    if terratorch_tmp_root:
        print(f"TERRATORCH_TMP_ROOT: {terratorch_tmp_root}")
    print()

    # Create log directory from the provided output_dir
    if verbose:
        print("Step 5: Setting up log directory...")
    base_log_dir = Path(args.output_dir).resolve()

    if not base_log_dir.exists():
        print(f"Error: Output directory does not exist: {base_log_dir}")
        sys.exit(1)

    log_dir = base_log_dir / execution_tag

    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {log_dir}")
    if verbose:
        print()

    # Validate venv_base_dir (mandatory)
    if verbose:
        print("Step 1: Validating virtual environment...")
    venv_path = Path(args.venv_base_dir)

    if not venv_path.exists():
        print(f"Error: Virtual environment does not exist: {venv_path}")
        sys.exit(1)

    # Check if tox is installed in the venv
    tox_path = venv_path / "bin" / "tox"
    if not tox_path.exists():
        print(f"Error: tox is not installed in the virtual environment: {venv_path}")
        print(f"Expected tox at: {tox_path}")
        sys.exit(1)

    if verbose:
        print(f"Virtual environment validated: {venv_path}")
        print(f"tox found at: {tox_path}")
        print()

    # Use current directory as full_path since tox will checkout the code
    full_path = Path.cwd()
    if verbose:
        print(f"Step 2: Working directory: {full_path}")
        print()

    # Test file path - validate it exists
    if verbose:
        print("Step 3: Validating test file...")
    test_file_path = Path(args.test_file)
    full_test_path = full_path / test_file_path

    if not full_test_path.exists():
        print(f"Error: Test file not found at {full_test_path}")
        print(f"Current directory: {Path.cwd()}")
        print(f"Full path: {full_path}")
        sys.exit(1)

    if verbose:
        print(f"Test file found: {full_test_path}")
        print()

    # Extract test names
    if verbose:
        print("Step 4: Extracting test cases...")

    # If single test specified, validate and use only that test
    tt_test_list = extract_test_names(full_test_path)

    # Determine which vLLM tests to run
    if args.test_name and args.test_name in VLLM_TESTS:
        vllm_tests_to_run = [args.test_name]
        tt_test_list = []
        if verbose:
            print(f"Running single vLLM test: {args.test_name}")
            print()
    elif args.test_name:
        # Single test specified but it's not a vLLM test
        vllm_tests_to_run = []
        if args.test_name in tt_test_list:
            tt_test_list = [args.test_name]
            if verbose:
                print(f"Running single test from file: {args.test_name}")
                print()
        else:
            print(f"Error: Test '{args.test_name}' not found")
            print(f"Available tests from file: {', '.join(tt_test_list)}")
            print(f"Available vLLM tests: {', '.join(VLLM_TESTS)}")
            sys.exit(1)
    else:
        # No single test specified, run all tests
        vllm_tests_to_run = VLLM_TESTS
        if verbose:
            print(f"Found {len(tt_test_list)} test cases:")
            for i, test_name in enumerate(tt_test_list, 1):
                print(f"  {i}. {test_name}")
            print()

    # Categorize tests
    if verbose:
        print("Step 6: Categorizing test cases...")
    models_fit_test = None
    dependent_test_list = []
    independent_test_list = []
    cleanup_test = None

    for test_name in tt_test_list:
        if test_name == PREREQUISITE_TEST:
            models_fit_test = test_name
        elif test_name == CLEANUP_TEST:
            cleanup_test = test_name
        elif test_name in DEPENDENT_TESTS:
            dependent_test_list.append(test_name)
        else:
            independent_test_list.append(test_name)

    # Determine which vLLM tests to run
    if args.test_name and args.test_name in VLLM_TESTS:
        vllm_tests_to_run = [args.test_name]
    elif args.test_name:
        # Single test specified but it's not a vLLM test, so no vLLM tests
        vllm_tests_to_run = []
    else:
        # No single test specified, run all vLLM tests
        vllm_tests_to_run = VLLM_TESTS

    total_tests = len(tt_test_list) + len(vllm_tests_to_run)
    if verbose:
        print(f"Categorized {total_tests} test(s):")
        print(f"  - Prerequisite: {1 if models_fit_test else 0}")
        print(f"  - Dependent: {len(dependent_test_list)}")
        print(
            f"  - Independent: {len(independent_test_list)} (from test file) + {len(vllm_tests_to_run)} (vLLM environments)"
        )
        print(f"  - Cleanup: {1 if cleanup_test else 0}")
        print()
    elif args.test_name:
        print(f"Validated environment - will run test: {args.test_name}")
    else:
        print(f"Validated environment and found {total_tests} tests to run")

    # Track dependent job IDs for cleanup dependency
    dependent_job_ids = []

    # Prepare activation command and environment variables
    activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
    user = os.environ.get("USER", "user")

    # Build environment variable export string for TERRATORCH_TMP_ROOT (used in main, not in submit_independent_tests)
    env_exports = ""
    if terratorch_tmp_root:
        env_exports = f" && export TERRATORCH_TMP_ROOT={terratorch_tmp_root}"

    # Track all submitted jobs for final summary table
    submitted_jobs = []

    if verbose:
        print("=" * 80)
        print("Step 7: Submitting jobs to LSF")
        print("=" * 80)
    else:
        print("Submitting jobs to LSF...")

    # Submit test_models_fit first
    if models_fit_test:
        verbose_msg = f"\n[1/4] Submitting prerequisite test: {models_fit_test}" if verbose else None
        models_fit_job_id = submit_single_test(
            test_name=models_fit_test,
            job_type="Prerequisite",
            verbose=verbose,
            user=user,
            activate_cmd=activate_cmd,
            branch_name=branch_name,
            env_exports=env_exports,
            python_version=python_version,
            log_dir=log_dir,
            submitted_jobs=submitted_jobs,
            dependent_job_ids=dependent_job_ids,
            gpu_config="num=1:mode=exclusive_process",
            verbose_message=verbose_msg,
            cleanup_venv=cleanup_venv,
            disable_pip_cache=disable_pip_cache,
        )

        # Submit dependent tests
        submit_dependent_tests(
            dependent_test_list,
            models_fit_test,
            models_fit_job_id,
            args.test_name,
            verbose,
            user,
            activate_cmd,
            branch_name,
            env_exports,
            python_version,
            log_dir,
            submitted_jobs,
            dependent_job_ids,
            cleanup_venv,
            disable_pip_cache,
        )

        # Submit cleanup test
        submit_cleanup_test(
            cleanup_test,
            dependent_job_ids,
            skip_cleanup,
            verbose,
            user,
            activate_cmd,
            branch_name,
            env_exports,
            python_version,
            log_dir,
            submitted_jobs,
            cleanup_venv,
            disable_pip_cache,
        )
    elif not args.test_name:
        # test_models_fit not found
        # In single test mode, this is OK - we'll run the test independently
        # In multi-test mode, show error only if there are dependent tests
        if verbose:
            print("\n" + "=" * 80)
            print("ERROR: test_models_fit not found in test suite")
            print("=" * 80)
        print(
            "Error: test_models_fit is a required prerequisite that creates checkpoints for dependent tests.",
            file=sys.stderr,
        )
        if verbose:
            print()

        if dependent_test_list and verbose:
            print(f"Warning: Skipping {len(dependent_test_list)} dependent test(s) (require test_models_fit):")
            for test_name in dependent_test_list:
                print(f"  - {test_name}")
            print()

    # Submit independent tests (including vLLM) - always executed
    # When in single test mode, dependent tests are treated as independent
    all_tests_to_submit = independent_test_list + (dependent_test_list if args.test_name else [])

    if all_tests_to_submit or vllm_tests_to_run:
        if verbose and not models_fit_test:
            print(f"Info: Submitting {len(all_tests_to_submit) + len(vllm_tests_to_run)} test(s) (including vLLM):")
        submit_independent_tests(
            all_tests_to_submit,
            vllm_tests_to_run,
            verbose,
            user,
            activate_cmd,
            branch_name,
            terratorch_tmp_root,
            python_version,
            log_dir,
            submitted_jobs,
            cleanup_venv,
            disable_pip_cache,
        )
    elif not models_fit_test and not args.test_name:
        print("\nError: No independent tests found. Cannot proceed without test_models_fit.")
        sys.exit(1)

    if verbose:
        print("\n" + "=" * 80)
        print("Job Submission Complete")
        print("=" * 80)

    # Save job IDs to file
    if submitted_jobs:
        jobs_file = log_dir / "job_ids.json"
        job_data = {
            "submission_time": datetime.now().isoformat(),
            "branch": branch_name,
            "execution_tag": execution_tag,
            "jobs": [
                {"type": job_type, "test_name": test_name, "job_id": job_id, "dependency": dependency}
                for job_type, test_name, job_id, dependency in submitted_jobs
            ],
        }

        with jobs_file.open("w") as f:
            json.dump(job_data, f, indent=2)

        if verbose:
            print(f"\nJob IDs saved to: {jobs_file}")

    # Print summary table of all submitted jobs
    if submitted_jobs:
        print("\nSubmitted Jobs Summary:")
        print("-" * 100)
        print(f"{'Type':<15} {'Test Name':<50} {'Job ID':<15} {'Depends On':<20}")
        print("-" * 100)
        for job_type, test_name, job_id, dependency in submitted_jobs:
            print(f"{job_type:<15} {test_name:<50} {job_id:<15} {dependency:<20}")
        print("-" * 100)
        print(f"Total jobs submitted: {len(submitted_jobs)}")
        print()

    print(f"\nLogs directory: {log_dir}")
    print(f"Check status with: python3 {sys.argv[0]} --check-status {log_dir}")
    print(f"Monitor jobs with: bjobs -J 'tt_{user}_*'")
    print("=" * 80)


if __name__ == "__main__":
    main()
