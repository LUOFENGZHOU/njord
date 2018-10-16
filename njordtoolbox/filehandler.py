#!/usr/bin/env python
# coding=utf-8

import os
import csv
import numpy
import torch
import pandas
import pickle

from random import randint

class FileHandler():
	"""Class that handles files.

	:attr cwd: the current working directory.
	:type cwd: str.
	:attr path: the path name.
	:type path: str.
	:attr home: the home location.
	:type home: str.
	"""

	def __init__(self, path="cwd"):
		"""Special method for class object construction.
		"""
		self.cwd = os.getcwd()
		self.home = self.get_home()
		if path == "cwd":
			self.path = os.getcwd()
		else:
			self.path = self.home
		self.counter = 1
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		msg = []
		msg.append("")
		msg.append("# File handler:")
		msg.append("# cwd = {}".format(self.cwd))
		msg.append("# home = {}".format(self.home))
		msg.append("# path = {}".format(self.path))
		msg.append("")		
		return "\n".join(msg)

	def __str__(self):
		"""Special method for class objet printable version.
		"""
		return self.__repr__()

	def path_append(self, *directory):
		"""Appends a directory to the path.
		"""
		path = []
		path.append(self.path)
		for name in directory:
			path.append(name)
		self.path = "/".join(path)
		self.check_path()
		return

	def path_append_with_random_number(self, directory, max=1000):
		"""Appends a numbered directory to the path.
		"""
		path = []
		path.append(self.path)
		path.append("{}_{}".format(directory, randint(0, max)))
		self.path = "/".join(path)
		self.check_path()
		return

	def path_append_with_number(self, directory, max=1000):
		"""Appends a numbered directory to the path.
		"""
		dir_list = os.listdir(self.path)
		number = 1
		for item in dir_list:
			if directory in item:
				if "{}_{}".format(directory, number) in dir_list:
					number += 1
				else:
					break
		path = []
		path.append(self.path)
		path.append("{}_{}".format(directory, number))
		self.path = "/".join(path)
		self.check_path()
		return

	def get_home(self):
		"""Returns the home path.
		"""
		if "HOME" in os.environ:
			home = os.environ["HOME"]
		elif os.name == "posix":
			home = os.path.expanduser("~/")
		elif os.name == "nt":
			if "HOMEPATH" in os.environ and "HOMEDRIVE" in os.environ:
				home = os.environ["HOMEDRIVE"] + os.environ["HOMEPATH"]
		else:
			home = os.environ["HOMEPATH"]
		return home

	def get_filename(self, filename, ext):
		"""Returns the filename.

		:param filename: the name of the file.
		:type filename: str.
		"""
		return "{}/{}.{}".format(self.path, filename, ext)

	def save_csv(self, filename, **kwargs):
		"""Save the item to a text file.

		:param filename: the name of the file.
		:type filename: str.		
		"""
		columns = []
		data = []
		for key in kwargs:
			columns.append(key)
			data.append(kwargs[key])
		df = pandas.DataFrame(columns=columns, data=[tuple(data)])
		name = self.get_filename(filename, "csv")
		if not os.path.exists(name):
			with open(name, "w") as file:
				df.to_csv(file, header=True, index=False, mode="w")
		else:
			with open(name, "a") as file:
				df.to_csv(file, header=False, index=False, mode="a")
		return

	def load_csv(self, filename):
		"""Load the filename.
		"""
		name = self.get_filename(filename, "csv")
		with open(name, "r") as file:
			df = pandas.read_csv(file)
		if "time" in df.columns:
			df = df.set_index("time")
			df.index = pandas.to_datetime(df.index)		
			return df
		else:
			return df

	def save_numpy(self, filename, item):
		"""Save a numpy item to a file.	
		"""
		name = self.get_filename(filename, "npy")
		with open(name, "wb") as file:
			numpy.save(file, item, allow_pickle=True)
		return True

	def load_numpy(self, filename):
		"""Load a numpy item from a file.
		"""
		name = self.get_filename(filename, "npy")
		with open(name, "rb" ) as file:
			item = numpy.load(file)	
		return item

	def save_pandas(self, filename, item):
		"""Save a pandas item to a file.
		"""
		file = self.get_filename(filename, "csv")
		item.to_csv(file)
		return True

	def load_pandas(self, filename):
		"""Load a pandas item from a file.
		"""
		file = self.get_filename(filename, "csv")
		return pandas.read_csv(file)

	def load_pandas_timeseries(self, filename):
		"""Load a pandas timeseries from a file.
		"""
		df = self.load_pandas(filename)
		if df.index.name == "time":
			pass
		else:
			df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)
		return df

	def save_pickle(self, filename, item):
		"""Save a pickleable item to a file.
		"""
		name = self.get_filename(filename, "pkl")
		with open(name, 'wb') as file:
			pickle.dump(item, file)
		return True

	def load_pickle(self, filename):
		"""Loads a pickleable item from a file.
		"""
		name = self.get_filename(filename, "pkl")		
		with open(name, 'rb') as file:
			item = pickle.load(file)
		return item

	def save_torch(self, filename, item):
		"""Save a torch item to a file.
		"""
		file = self.get_filename(filename, "txt")
		torch.save(item, file)
		return True

	def load_torch(self, filename):
		"""Load a torch item from a file.
		"""
		file = self.get_filename(filename, "txt")
		item = torch.load(file, map_location="cpu")
		return item

	def load_dataframe(self, filename):
		"""Load the dataframe for the specified filename

		:param filename: the name of the file.
		:type filename: str.

		:return: the dataframe.
		:rtype: pandas dataframe.
		"""
		df = pandas.read_csv(filename)
		df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)
		return df

	def check_path(self):
		"""Check if the current path exists, if not creates one.
		:return: True if the path exists, False otherwise and creates it.
		:rtype: bool.
		"""
		if not os.path.exists(self.path):
			os.makedirs(self.path)
			return False
		else:
			return True
