#!/usr/bin/env python
# coding=utf-8

import time
import numpy
import pandas
import datetime

# -------------------------------------------- #

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
		"""
		columns = df.columns
		for i in range(0,len(columns)):
			old = columns[i]
			new = "{}_{}".format(old, marker)
			df = df.rename(columns={old:new})
		return df

	def select(self, df, marker):
		"""Select a sub dataframe for the specified marker.
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
				df = 100 * ( df / self.root - 1.0 )
			elif method == "pct":
				df = 100 * ( df / df.shift(lk+1) - 1.0 )
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
				df = 100 * ( df / self.root[target] - 1.0 )
			elif method == "pct":
				df = 100 * ( df / df.shift(la+1) - 1.0 )
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

	# ---------------------------------------------- #
	# --- 1. Prometheus timeseries basic methods --- #
	# ---------------------------------------------- #

	def set_target(self, target):
		"""Set the target.

		:param targets: the name of the target(s).
		:type targets: str, tuple(str), list<str>.
		"""
		self.targets = None
		for item in self.root.columns:
			if target in item:
				self.targets = (item,)
				break
		if self.targets is None:
			raise ValueError("Please provide a consistent targets.")
		return 

	def set_features(self, features=None):
		"""Set the features.

		:param features: the name of the features (optional).
		:type features: tuple(str) or lst<str>.
		"""
		self.features = tuple(self.root.columns)
		return

	def _get_timeseries_label_features(self, marker="@X"):
		features = []
		for item in self.timeseries.columns:
			if marker in item:
				features.append(item)
		return features

	def _get_timeseries_label_targets(self, marker="@Y"):
		targets = []
		for item in self.timeseries.columns:
			if marker in item:
				targets.append(item)
		return targets

	def _get_timeseries_shifted_single_elem(self, df, step, marker=""):
		columns = df.columns
		df = df.shift(step)
		for i in range(0,len(columns)):
			old = columns[i]
			new = "{}_{}_{}".format(marker, old, step)
			df = df.rename(columns={old:new})
		return df

	# ------------------------------------------------ #
	# --- 3. Prometheus timeseries dataset methods --- #
	# ------------------------------------------------ #

	def _get_timeseries_features_as_dataframe(self):
		features = self._get_timeseries_label_features()
		return self.timeseries.loc[:,features]

	def _get_timeseries_targets_as_dataframe(self):	
		targets = self._get_timeseries_label_targets()
		return self.timeseries.loc[:,targets]

	def _get_timeseries_features_as_array(self):
		features = self._get_timeseries_features_as_dataframe()
		features = features.values
		(m, n) = numpy.shape(features)
		features = features.reshape(m, len(self.past), int(n/len(self.past)))
		return features

	def _get_timeseries_targets_as_array(self):
		targets = self._get_timeseries_targets_as_dataframe()
		targets = targets.values
		(m, n) = numpy.shape(targets)
		if len(self.future) > 1:
			targets = targets.reshape(m, len(self.future), int(n/len(self.future)))
		return targets

	def get_dataset(self):
		"""Return the dataset for the current root.

		:param balanced: balance the dataset accross the categories.
		:type balanced: bool.

		:return X_train: the features.
		:rtype X_train: numpy array of shape (m, t_x, n_x).
		:return Y: the targets.
		:rtype Y: numpy array of shape (m, n_y).
		:return t: the timeseries index.
		:rtype t: numpy array of shape (m, 1).
		"""
		
		# get the features
		X = self._get_timeseries_features_as_array()
		
		# get the targets
		Y = self._get_timeseries_targets_as_array()

		# remove timeseries
		self.timeseries = None

		return (X, Y)
