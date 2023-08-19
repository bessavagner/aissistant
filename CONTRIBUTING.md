# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

## Types of Contributions

### Report Bugs

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

You can never have enough documentation! Please feel free to contribute to any
part of the documentation, such as the official docs, docstrings, or even
on the web in blog posts, articles, and such.

### Submit Feedback

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started!

Ready to contribute? Here's how to set up `aissistant` for local development.

1. Fork the repository, clone and add remote:
   
```bash
git clone --recursive https://github.com/[your-user]/aissistant.git # TODO
git remote add upstream https://github.com/bessavagner/aissistant.git # TODO
```

2. Install `aissistant` using `poetry`:

```bash
$ poetry install
```

If you don't have poetry installed:

```bash
pip install poetry
```

Note that currently `aissistant` supports Python 3.11.

3. Use `git` (or similar) to create a branch for local development and make your changes:

```bash
git checkout -b branch_name
```

4. Set `DEBUG = True` at `settings.py` while developing. This enables debugging logs that help developers or clients see points where the package or application needs improvement or some kind of attention. Note that `settings.py` has some custom loggers. Use `client` or `standard` for the package. For app, cli or gui user `debbug`, `report` or `report_debug`. If your contribution has some points you judge is worth logging and info or warning message, prefer nest it in a DEBUG switch:

```python
if DEBUG:
    logger.warning('Log text')
```
5. When you're done making changes, check that your changes conform to any code formatting requirements and pass any tests.
   
6. Add and commit your changes. Prefer using [Angular commit style](https://github.com/angular/angular.js/blob/master/DEVELOPERS.md#commit-message-format):

```bash
git commit -m -b <type>(<scope>): <subject>
```
- `<type>` should be one of the following: feat, fix, docs, style, refactor, test, or chore.
- `<scope>` is optional and should be a brief description of the affected component or module.
- `<subject>` should be a short, descriptive message of the changes you made.
    For example: feat(login): add forgot password link.

We intent to in the future to use [Python Semantic Release (PSR)](https://python-semantic-release.readthedocs.io/en/latest/), so prefer to write your commits with `<type>` as one of:

 - **feat**: A new feature.
 - **fix:** A bug fix.
 - **docs**: Documentation changes.
 - **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc).
 - **refactor**: A code change that neither fixes a bug nor adds a feature.
 - **perf**: A code change that improves performance.
 - **test**: Changes to the test framework.
 - **build**: Changes to the build process or tools.


6. Push your changes to the remote branch.
 - Open a pull request (PR) against the main branch and describe the changes you made.
 - Wait for review and address any feedback.
 - Once your changes are approved, they will be merged into the main branch.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include additional tests if appropriate.
2. If the pull request adds functionality, the docs should be updated.
3. The pull request should work for all currently supported operating systems and versions of Python.

## Code of Conduct

Please note that the `aissistant` project is released with a
Code of Conduct. By contributing to this project you agree to abide by its terms.
