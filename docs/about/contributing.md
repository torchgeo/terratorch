# Contributing to TerraTorch

Contributions to an open source project can came in different ways, but we could summarize them in three main
components: adding code (as new models, tasks and
auxiliary algorithms or even addressing the solution of a bug), examples using the software (scripts, yaml files and notebooks showcasing the package) and documentation.
All these ways are valid for TerraTorch
and the users are welcome to contribute in any of these fronts. However, some recommendations and rules are
necessary in order to facilitate and organize the process. And this is the matter of the next paragraphs. 

## Contributing with code

It is not a trivial task to determine how a modification in the source code will impact already implemented
and established features, in this way, for any modification in the core source code (`terratorch/`) we
automatically execute
a pipeline with hundreds of unit and integration tests to verify that the package have not broken after the
modification be merged to `main`. In this way, when an user wants to modify
`terratorch` for adding new features or bufixes, this are the best practices. 

## General steps

* Clone the repository, if you were granted with direct writing access:
    ```
        git clone git@github.com:IBM/terratorch.git
    ```
    otherwise, clone it from your personal fork.
* This repository uses `pre-commit`, a tool which automatically runs basic
    steps before sending modifications to the remote (as linting, for example).
    See how to configure it [here](https://pre-commit.com/#installation). 
* If you are an user outside the IBM org, create a fork to add your modifications. If you are inside the IBM
    org or have received writing provileges, prefer to create a branch for it. 
* If you are adding new features, we ask you to also add tests for it. These tests are defined in the
    directory `tests/` and are fundamental to check if your feature is working as expected and not breaking
    anything. If your feature is something more complex, as a new model or auxiliary algorithm, you can also
    (optionally) to add a complete example, as a notebook or Python script, demonstrating how the feature works.
* After finishing your modifications, we recommend you to test locally using `pytest`, for example:
    ```
    pytest -s -v tests/
    ```

## Reproducing the CI test environment

CI resolves test dependencies against a pinned constraints file so that the exact same
versions install everywhere (issue [#912](https://github.com/torchgeo/terratorch/issues/912)).
To reproduce that environment locally, install with the same constraints:

```
pip install -e .[test] -c constraints/test-py313.txt
```

The constraints only *cap* versions, so extras that are not part of the lock (for
example `wxc` or `geobenchv2`) still install and simply float their extra dependencies.

The file is generated from `pyproject.toml` for the CI Python version (3.13). Regenerate
it whenever you change dependencies, and commit the result:

```
scripts/compile_constraints.sh   # requires `uv` and Python 3.13
```

## Testing using `tox`

* If you want to run tests for more than one Python version, you can use `tox`:
    1. install `tox` (via `pip` or `uv`)
    2. Run `tox`:
        ```
        tox run
        ```
    3. For a specific version:
        ```
        tox run run -e 3.13
        ```
    4. In case you don't have some Python versions installed, `tox` will skip them. 

## Testing with `podman` containers

* Alternatively, it is possible to run the tests inside a container environment using `podman`. All the modifications done for the files inside the repository will be copied to the image and the tests will be updated. To do it, proceed in the following way:

    1. Install `podman` in your system.
    2. Building the base images:
        ```
        ./run_base_images_build.sh 
        ```
    3. Running the tests inside each environment:
        ```
        ./run_tests_container.sh 
        ```
    4. The images will remain in your system to be reused and will be rebuilt just
       when some major change is done in the source code. 

## Other recommended practices

* If all the tests are passing, you can open a PR to `terratorch:main` describing what you are adding and why
    that is important to be merged. You
    do not need to choose a reviewer, since the maintainers will check the new open PR and request review for it by themselves.  
* The PR will pass through the tests in GitHub Actions and if the reviewer approve it, it will soon be merged. 
* It is recommended to add a label to your PR. For example `bug`, when it solves some issue or `enhancement`
    when it adds new features. 
* It is recommended to periodically rebase commit history of your
    branch by squashing incremental commits. It will help to keep the PR clean and better organized.
    To do it, run:
    ```
    git log
    ```
    And check the number of commits you want to aggregate.
    So:
    ```
    git rebase -i HEAD~<number of commits you want to rebase>
    ```
    In the editor, replace the `pick` with `squash` in front of each commit you
    want to hide. Just do it for the line `2` onwards. The first commit will
    became the umbrella commit and the other will became sub-commits. 
    Save and push it to remote using:
    ```
    git push -f origin <branch>
    ```
!!! caution
    Your PR will not be merged if the automatic tests are failing and the user which has sent the PR is responsible for fixing it. 

## Contributing with documentation

Documentation is core for any project, however, most part of the time, the developers do not have the time (or
patience) to carefully document all the codebase, in this way, contributions from interested users are always
welcome. 
To add documentation to TerraTorch, you need to be familiar with Markdown, a clean markup language, and the pages are built with
MkDocs, a framework which relies on Markdown in order to create webpages as this you are reading. The documentation is automatically
deployed by a bot just after the push or merge to the branch dedicated to store the latest documentation.
These are the steps to configure your environment to work with documentation.

* Clone the documentation branch to your machine:
```
    git fetch origin docs
    git checkout docs
    git pull origin docs
```
* Add your modifications and open a PR to the branch `docs`. It is recommended to add the label `documentation` to your PR.
* The PR will be reviewed and approved if it is considered relevant by the maintainers. 

## Contributing by reporting issues

The users also can contribute by reporting issues they observed during their experiments. When reporting an issue, provide the more details as possible about the problem, as the configuration of your system,
the terminal output and even the files required to reproduce it. To
illustrate it, take a
look on an example of [a very well-written issue](https://github.com/IBM/terratorch/issues/506). 
