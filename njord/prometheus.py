#!/usr/bin/env python
# coding=utf-8

import time
import numpy
import pandas
import datetime

# -------------------------------------------- #

class Prometheus:
	"""Class that handles timeseries datasets.

	This class is suited for financial data such as price, trades, ohlc, spread ask and bid.
	The root is a pandas dataframe, from the root it is possible to build training examples
	suited for machine learning applications. The features (X) are the training examples and
	the targets (Y) are the values to predict. They can be prices or rise and drop categories.
	The user should specify the past as the horizon window to build the training examples. The future
	specifies the number of steps ahead to look for such as to build the targets. 

	:attr root: a standard timeseries dataframe.
	:type root: pandas dataframe.
	:attr past: the past horizon window to build the features.
	:type past: list<int>.
	:attr future: the future horizon window to build the targets.
	:type future: list<int>.
	:attr ewma_span: optional attribute for exponential moving average data processing.
	:type ewma_span: int.
	:attr ewma_freq: optional attribute for exponential moving average data sampling.
	:type ewma_frea: str.
	:attr categories: optional attribute such as to build drop and rise categories for classification purpose.
	:type categories: tuple(tuple(lower_bound, upper_bound), ...).
	:attr features: name of the features that are represented in the root.
	:type features: list<str>.
	:attr _norm: normalisation methods.
	:type _norm: dict.
	"""

	def __init__(self):
		"""Special method for class object construction.
		"""
		self.root = None
		self.timeseries = None
		self.past = None
		self.future = None		
		self.targets = None
		self.features = None
		self.norm = None
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = []
		_repr.append("")
		_repr.append("# --- Prometheus --- #")
		_repr.append("# past .......... = {}".format(self.past))
		_repr.append("# future ........ = {}".format(self.future))
		_repr.append("# targets ....... = {}".format(self.targets))		
		_repr.append("# features ...... = {}".format(self.features))	
		_repr.append("# norm .......... = {}".format(self.norm))
		_repr.append("# ------------------- #")
		_repr.append("")
		return "\n".join(_repr)

	def __str__(self):
		"""Special method for class object printable version.
		"""
		return self.__repr__()

	def set_attr(self, attr):
		"""Set the attributes of prometheus from a dictionnary.

		:param attr: the attributes.
		:type attr: dict.
		"""
		self.past = attr["past"]
		self.future = attr["future"]
		self.targets = attr["targets"]			
		self.features = attr["features"]
		self.norm = attr["norm"]
		return

	def get_attr(self):
		"""Returns the attributes of prometheus as a dictionnary.

		:return: the attributes.
		:rtype: dict.
		"""
		attr = {}
		attr["past"] = self.past
		attr["future"] = self.future
		attr["targets"] = self.targets
		attr["features"] = self.features
		attr["norm"] = self.norm
		return attr

	# ----------------------------------- #
	# --- 0. Prometheus basic methods --- #
	# ----------------------------------- #

	def _check_dataframe(self, df):
		if df.index.name == "time":
			pass
		else:
			df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)		
		return df

	def set_root(self, df):
		"""Set the root with the specified dataframe.

		:param df: the dataframe to set as root.
		:type df: pandas dataframe.

		.. note:: df should have it's index or a column specifying the timestamp labeld as 'time'.
		"""
		if df.index.name == "time":
			pass
		else:
			df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)		
		self.root = df
		return

	def set_past(self, past):
		"""Set the past horizon as a list.

		:param past: length of the past.
		:type past: int.
		"""
		self.past = list(range(0,past))[::-1]
		print("Prom past = {}".format(self.past))
		return

	def set_future(self, future):
		"""Set the future horizon as a negative integer.

		:param past: length of the future.
		:type past: int.
		"""
		if isinstance(future, int):
			self.future = []
			for i in range(1, future+1):
				self.future.append(-i)
			print("Prom future = {}".format(self.future))
		else:
			raise ValueError("Prometheus exception, please provide consistent future.")
		return

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

	def set_norm(self, method, ref=None, marker="#t"):
		"""Set the normalisation method.

		:param method: the name of the method.
		:type method: str.
		:param ref: the normalisation reference (optional).
		:type ref: str.
		"""
		self.norm = {"method":method, "ref":ref, "marker":marker}
		return

	# ---------------------------------------------- #
	# --- 1. Prometheus timeseries basic methods --- #
	# ---------------------------------------------- #

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

	def _get_normalise_column(self, df, marker, column=None):
		labels = []
		for item in df.columns:
			if marker in item:
				if column is None:
					labels.append(item)
				else:
					if column in item:
						labels.append(item)
					else:
						pass
		return labels

	def _normalise_pivot(self, df, ref, marker):
		df_norm_columns = self._get_normalise_column(df, marker)
		ref_norm_columns = self._get_normalise_column(self.root, marker, ref)
		df.loc[:,df_norm_columns] = 100 * ( 
			df.loc[:,df_norm_columns].values 
			/ self.root.loc[:,ref_norm_columns].values 
			- 1.0 )
		return df

	def _normalise_pct_change(self, df, ref, marker):
		df_norm_columns = self._get_normalise_column(df, marker)
		ref_norm_columns = self._get_normalise_column(df, marker, ref)
		#print(ref_norm_columns)
		df.loc[:,df_norm_columns] = 100 * ( 
			  df.loc[:,df_norm_columns].values 
			/ df.loc[:,ref_norm_columns].shift(1).values 
			- 1.0 )
		return df

	def _normalise_log_change(self, df, ref, marker):
		df_norm_columns = self._get_normalise_column(df, marker)
		ref_norm_columns = self._get_normalise_column(df, marker, ref)
		df.loc[:,df_norm_columns] = 100 * numpy.log(
			  df.loc[:,df_norm_columns].values
			/ df.loc[:,ref_norm_columns].shift(1).values )
		return df

	def set_timeseries(self):
		"""Set the timeseries dataframe for the specified targets and features.
		"""

		# init the timeseries
		self.timeseries = pandas.DataFrame()

		# set the features of the timeseries
		if self.features is not None:
			
			for step in self.past:

				# get the shifted dataframe
				df_shift = self._get_timeseries_shifted_single_elem(
					df=self.root.loc[:,self.features], step=step, marker="@X")

				# normalise the new dataframe
				if self.norm["method"] is None:
					pass
				elif self.norm["method"] == "pct_change":
					df_shift = self._normalise_pct_change(
						df_shift, self.norm["ref"], self.norm["marker"])
				elif self.norm["method"] == "pvt_change":
					df_shift = self._normalise_pivot(
						df_shift, self.norm["ref"], self.norm["marker"])
				elif self.norm["method"] == "log_change":
					df_shift = self._normalise_log_change(
						df_shift, self.norm["ref"], self.norm["marker"])					
				else:
					raise ValueError(
						"Please provide a consistent normalisation method.")

				# add the dataframe to the timeseries
				self.timeseries = pandas.concat([self.timeseries, df_shift], 
					axis=1, join="outer")

		# set the targets of the timeseries
		if self.targets is not None:

			for step in self.future:

				# get the shifted dataframe
				df_shift = self._get_timeseries_shifted_single_elem(df=self.root.loc[:,self.targets], step=step, marker="@Y")

				# normalise the new dataframe
				if self.norm["method"] is None:
					pass
				elif self.norm["method"] == "pct_change":
					df_shift = self._normalise_pct_change(
						df_shift, self.targets[0], self.norm["marker"])
				elif self.norm["method"] == "pvt_change":
					df_shift = self._normalise_pivot(
						df_shift, self.targets[0], self.norm["marker"])
				elif self.norm["method"] == "log_change":
					df_shift = self._normalise_log_change(
						df_shift, self.targets[0], self.norm["marker"])
				else:
					raise ValueError("Please provide a consistent normalisation method.")

				# add the dataframe to the timeseries
				self.timeseries = pandas.concat([self.timeseries, df_shift], axis=1, join="outer")
				
		# clean the timeseries
		self.timeseries = self.timeseries.dropna(axis=0, how="any")

		return

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
