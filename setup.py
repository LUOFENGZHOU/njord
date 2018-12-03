#!/usr/bin/env python
# coding=utf-8

from setuptools import setup

setup(name="njord",
		version="0.2",
		description="njord project: package for datasience and cryptotrading",
		url="https://github.com/njord-project/njord",
		author="mdhoffschmidt",
		author_email="mdhoffschmidt@protonmail.com",
		license=None,
		packages=["njord"],
		install_requires=["numpy", "pandas"],
		zip_safe=False)