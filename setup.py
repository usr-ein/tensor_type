#!/usr/bin/env python3
import pathlib

from setuptools import setup

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="tensor_type",
    version="0.1.0",
    packages=["tensor_type"],
    description="Annotates shapes of PyTorch Tensors using type annotation in Python3, and provides optional runtime shape validation.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/sam1902/tensor_type",
    author="Samuel Prevost",
    author_email="samuel.prevost@pm.me",
    licence="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
    install_requires=['torch'],
)
