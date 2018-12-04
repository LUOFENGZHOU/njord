#!/usr/bin/env python
# coding=utf-8

import time
import numpy
import pandas
import datetime


class Batch:
	"""Class that handles timeseries datasets.

	:param root: the root dataset.
	:type root: pandas dataframe.
	:param data: the batch dataset.
	:type data: pandas dataframe.
	:param lk: the lookback horizon.
	:type lk: int.
	:param la: the lookahead horizon.
	:type la: int.
	"""

	def __init__(self, root, lk, la):
		"""Special method for class object construction.

		:param root: the root dataset.
		:type root: pandas dataframe.
		:param lk: the lookback horizon.
		:type lk: int.
		:param la: the lookahead horizon.
		:type la: int.
		"""
		self.root = self._check_df(root)
		self.lk = self._check_lk(lk)
		self.la = self._check_la(la)
		self.data = None	
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {}
		_repr["root"] = self.root.size
		_repr["lk"] = self.lk
		_repr["la"] = self.la
		if self.data is not None:
			_repr["data"] = self.data.size
		else:
			_repr["data"] = None
		return _repr

	def __str__(self):
		"""Special method for class object printable version.
		"""
		msg = []
		for key, item in self.__repr__().items():
			msg.append("{} = {}".format(key, item))
		return "{}({})".format(self.__class__.__name__, ", ".join(msg))

	def _check_df(self, df):
		"""Check the dataframe.
		"""
		if df.index.name == "time":
			pass
		else:
			df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)		
		return df

	def _check_lk(self, lk):
		"""Check the lookback horizon.
		"""
		if isinstance(lk, int):
			return list(range(0,lk))[::-1]
		else:
			raise TypeError()

	def _check_la(self, la):
		"""Check the lookahead horizon.
		"""
		if isinstance(la, int):
			index = []
			for i in range(1, la+1):
				index.append(-i)
			return index
		else:
			raise TypeError()

	def sign(self, df, marker):
		"""Sign the dataframe columns with a marker.

		:param df: the dataframe whose columns must be signed.
		:type df: pandas DataFrame.
		:param marker: the signature.
		:type marker: str.

		:return: the dataframe with signed columns.
		:rtype: pandas DataFrame.
		"""
		columns = df.columns
		for i in range(0,len(columns)):
			old = columns[i]
			new = "{}_{}".format(old, marker)
			df = df.rename(columns={old:new})
		return df

	def select(self, df, marker):
		"""Select a sub dataframe for a specified marker.

		:param df: the dataframe.
		:type df: pandas DataFrame.
		:param marker: the marker.
		:type marker: str.

		:return: dataframe. 
		:rtype: pandas DataFrame.		
		"""
		columns = []
		for col in df.columns:
			if marker in col:
				columns.append(col)
		return df.loc[:,columns]

	def build(self, target=None, method="pvt"):
		"""Build and return the batch.
		"""

		# Initialise an empty batch.
		batch = []

		# Check the target.
		if target is None:
			target = list(self.root.columns)
		elif isinstance(target, str):
			target = [target]
		else:
			pass
		
		# Set the features.
		for lk in self.lk:
			
			# Shift the root.
			df = self.root.shift(lk)

			# Normalise the shifted dataframe.
			if method == "pvt":
				df = df / self.root - 1.0
			elif method == "pct":
				df = df / self.root.shift(lk+1) - 1.0
			else:
				pass

			# Sign the new dataframe.
			df = self.sign(df, "@X_{}".format(lk))

			# Append the dataframe to the batch.
			batch.append(df)

		# Set the targets.
		for la in self.la:

			# Shift the root.
			df = self.root[target].shift(la)

			# Normalise the shifted dataframe.
			if method == "pvt":
				df = df / self.root[target] - 1.0
			elif method == "pct":
				df = df / self.root.shift(la+1) - 1.0
			else:
				pass

			# Sign the new dataframe.
			df = self.sign(df, "@Y_{}".format(la))

			# Append the dataframe to the batch.
			batch.append(df)

		# Concatenate all the frames to build the batch.
		batch = pandas.concat(batch, axis=1, join="outer")

		# Clean the batch from nans. 
		batch = batch.dropna(axis=0, how="any")		

		# Select the features.
		X = self.select(batch, "@X").values
		(m, n) = numpy.shape(X)
		X = X.reshape(m, len(self.lk), int(n/len(self.lk)))		

		# Select the targets.
		Y = self.select(batch, "@Y").values
		(m, n) = numpy.shape(Y)
		Y = Y.reshape(m, len(self.la), int(n/len(self.la)))	

		return X, Y
