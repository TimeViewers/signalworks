#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import find_packages, setup

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
        "Programming Language :: Python :: 3.7",
    ],
    description="Library to handle signal data and perform signal processing computations",
    entry_points={"console_scripts": ["signalworks=signalworks.cli:main"]},
    install_requires=["numpy", "scipy", "numba==0.42.0", "llvmlite==0.27.0"],
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="signalworks",
    name="signalworks",
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    test_suite="tests",
    url="https://github.com/lxkain/signalworks",
    version="0.1.7",
    zip_safe=False,
)
