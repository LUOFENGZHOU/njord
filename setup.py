#!/usr/bin/env python
# coding=utf-8

from setuptools import setup

setup(name="njord",
	version="0.1",
	description="njord project: package for machine learning and cryptotrading",
	url="https://github.com/njord-project/njord",
	author=["mdhoffschmidt", "evrardts"],
	author_email="mdhoffschmidt@protonmail.com",
	license="Mit License",
	packages=["njord"],
	install_requires=["numpy", "pandas"],
	zip_safe=False)