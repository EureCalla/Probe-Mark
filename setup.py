#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    author="FangYu Lin",
    author_email="hatdog63@gmail.com",
    name="probe_mark",
    keywords="probe_mark",
    packages=find_packages(
        include=["probe_mark", "probe_mark.*"]
    ),
    test_suite="tests",
    license="MIT license",
    version="0.1.0",
)
