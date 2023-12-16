#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='trainer',
    version='0.0.1',
    description='LM training',
    author='',
    author_email='',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/ipieter/trainer',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

