#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import os
from setuptools import setup, find_packages
from typing import List


def read(fname: str) -> List[str]:
    contents = open(os.path.join(os.path.dirname(__file__), fname)).read()
    return contents.split("\n")


def read_requirements(fname: str) -> List[str]:
    contents = read(fname)[1:]
    for index, requirement in enumerate(contents):
        if requirement.startswith("git+"):
            _, _, pkg_name = requirement.rpartition("=")
            contents[index] = pkg_name + "@" + requirement
    return contents


with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

setup(
    author="Alex Kain",
    author_email="lxkain@gmail.com",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    description="Library to handle signal data and perform signal processing computations",
    entry_points={"console_scripts": ["signalworks=signalworks.cli:main"]},
    install_requires=read_requirements("requirements.txt"),
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="signalworks",
    name="signalworks",
    packages=find_packages(include=["signalworks"]),
    setup_requires=["pytest-runner"],
    test_suite="tests",
    tests_require=read_requirements("requirements-dev.txt"),
    url="https://github.com/lxkain/signalworks",
    version="0.1.0",
    zip_safe=False,
)
