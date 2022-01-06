#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name='irtorch',
    version='2.1',
    description='Item Response Theory Parameter Estimator',
    author='nakt',
    author_email='nakt@walkure.net',
    install_requires=['numpy', 'pandas', 'torch', 'pytorch_lightning', 'fire'],
    python_requires='>=3.7',
    packages=find_packages(exclude=('tests', 'docs')),
    test_suite='tests'
)
