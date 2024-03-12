from distutils.core import setup

from setuptools import find_packages

setup(
    name="benchmark",
    packages=find_packages(exclude=("*test*",)),
)
