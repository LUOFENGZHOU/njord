#!/usr/bin/env python
# coding=utf-8

import random
from .record import Record


class History:
	"""Class that handles historical data manipulation.

	:attr df: the historical data.
	:type df: pandas DataFrame.
	:attr name: the name of the history.
	:type name: str.
	:attr period: the period of the data in seconds.
	:type period: int.
	:attr timestamp: the current timestamp.
	:type timestamp: numpy datetime64.		
	:attr timestamp_first: the first timestamp of the history.
	:type timestamp_first: numpy datetime64.
	:attr timestamp_last: the first timestamp of the history.
	:type timestamp_last: numpy datetime64.
	:attr length: the length of the history.
	:type length: int.
	:attr record: the recorded historical data.
	:type record: dict of Record.
	"""

	def __init__(self, df, name="history"):
		"""Special method for class object construction.

		:param df: the history dataframe.
		:type df: pandas dataframe.
		:param name: the name for the history object.
		:type name: str.
		"""
		
		self.df = df
		self.name = name
		self.period = self._get_period()
		self.columns = df.columns

		# Set the first and last timestamp of the history.
		self.timestamp_first = self.df.index.values[0]
		self.timestamp_last = self.df.index.values[-1]	

		# Set the initial timestamp.
		self.timestamp = self.df.index.values[0]

		# Set the scope records as empty.		
		self.records = {}

		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {}
		_repr.update({"name": self.name})
		_repr.update({"period": self.period})
		_repr.update({"timestamp": self.timestamp})
		_repr.update({"timestamp_first": self.timestamp_first})
		_repr.update({"timestamp_last": self.timestamp_last})
		return _repr

	def __str__(self):
		"""Special method for class object printable version.
		"""
		_str = []
		for a, b in self.__repr__().items():
			_str.append("{} = {}".format(a, b))
		return "{}({})".format(self.__class__.__name__, ", ".join(_str))

	def __len__(self):
		"""Special method for class object length.
		"""
		return len(self.df)

	def clear(self):
		"""Clear the records.
		"""
		for key in self.records:
			self.records[key].clear()
		return

	def _get_period(self):
		"""Returns the dataframe period in seconds.
		"""
		delta = self.df.index.values[1] - self.df.index.values[0]
		period_nano_seconds = float(delta)
		period_seconds = int(period_nano_seconds / 1.0E+9)
		return period_seconds

	def randomtimestamp(self):
		random_index = random.randrange(len(self.df)-1000)
		return self.df.index.values[random_index]

	def get_list_values(self):
		"""Returns the current data values as a list.

		:return: the data values for the current timestamp.
		:rtype: list.
		"""
		try:
			return list(self.df.loc[self.timestamp,:])
		except KeyError:
			return None

	def get_dict_values(self):
		"""Returns the current data values as a dict.

		:return: the data values for the current timestamp.
		:rtype: dict.
		"""
		try:
			return dict(self.df.loc[self.timestamp,:])
		except KeyError:
			return None

	def get_array_values(self):
		"""Returns the current data values as an array.

		:return: the data values for the current timestamp.
		:rtype: numpy array.
		"""
		try:
			return self.df.loc[self.timestamp,:].values
		except KeyError:
			return None			

	def get_values_by_name(self, name, look=0):
		"""Returns the current data values for the given name.

		:param names: the requested name
		:type names: list<str>.
		:param look: steps to look from current timestamp.
		:type look: int.

		:return: the data values for the current timestamp.
		:rtype: float.		
		"""
		try:
			timestamp = self.timestamp + int(look * self.period * 1.0E+9)
			dict_val = dict(self.df.loc[timestamp,:])
			for key in dict_val:
				if name in key:
					return dict_val[key]
		except KeyError:
			return None

	def get_values_by_names(self, *names):
		"""Returns the current data values for the given names.

		:param names: the list of names requested.
		:type names: list<str>.

		:return: the data values for the current timestamp.
		:rtype: dict.		
		"""
		try:
			dict_val = dict(self.df.loc[timestamp,:])
			val = {}
			for name in names:
				for key in dict_val:
					if name in key:
						val[name] = dict_val[key]
			return val
		except KeyError:
			return None

	def step(self, seconds=None):
		"""Increment the current timestamp by the specified seconds.

		:param seconds: the number of seconds to increment (optional).
		:type seconds: int or float.
		"""
		if seconds is None:
			seconds = self.period
		self.timestamp += int(seconds * 1.0E+9)
		self.register()
		return

	def scope(self, *names):
		"""Scope the names during the step.

		:param names: the names to be scoped.
		:type names: list.
		"""
		for name in names:
			if name not in self.records.keys():
				self.records[name] = Record(name)
		return

	def register(self):
		"""Register the requested information.
		"""
		for key in self.records:
			val = self.get_values_by_name(key)
			if val is None:
				pass
			elif isinstance(val, float):
				self.records[key].append(self.timestamp, val)
			elif isinstance(val, list):
				self.records[key].append(self.timestamp, val[0])
			else:
				pass			
		return

