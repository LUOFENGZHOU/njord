#!/usr/bin/env python
# coding=utf-8

from setuptools import setup


setup(name="njordtoolbox",
		version="0.2",
		description="toolbox for the njord project",
		url="https://github.com/mdhoffschmidt/njordtoolbox",
		author="mdhoffschmidt",
		author_email="mdhoffschmidt@protonmail.com",
		license=None,
		packages=["njordtoolbox"],
		install_requires=["numpy", "pandas", "torch"],
		zip_safe=False)