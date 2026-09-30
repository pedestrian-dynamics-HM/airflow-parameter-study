#!/usr/bin/env python3

# References
# [1] https://stackoverflow.com/a/67692

import importlib.util
import os

from setuptools import find_packages, setup

# Package
PACKAGE_NAME = "airflow-uq"
AUTHOR = "Sophia Wagner, Simon Rahn"
AUTHOR_EMAIL = "sophia.wagner@hm.edu"
URL = "https://gitlab.lrz.de/vadere/airflow-uq"
LICENSE = "LGPL"
DESCRIPTION = "Uncertainty quantification with the Vadere airflow-exposure model"
PY_REQUIRES = ">=3.8"


def version():
    """Reads the version from _version.py without importing parts of
    suqc (which would require some of the dependencies already installed)."""
    # code parts taken from [1]

    path2setup = os.path.dirname(__file__)
    version_file = os.path.join(path2setup, "vimuq", "__init__.py")
    version_file = os.path.abspath(version_file)

    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.__version__


def requirements():
    with open("requirements.txt") as f:
        ret = [l.strip() for l in f.readlines()]
    return ret


def readme():
    with open("README.md") as f:
        return f.read()


# Writes a file that gives information about the version such that
# "infection-model-uq.__version__" provides the current version, which is a
# convention in Python:
path_package_indicator_file = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "PACKAGE.txt"
)


setup(
    name=PACKAGE_NAME,
    version=version(),
    description=DESCRIPTION,
    long_description=readme(),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(exclude=["tests"]),
    # install_requires=requirements(),  # this works only if requirements.txt
    # does not contain relative paths, e.g.,
    # "-e ./suqc"
    license=LICENSE,
    python_requires=PY_REQUIRES,
)
