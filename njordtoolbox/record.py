#!/usr/bin/env python
# coding=utf-8

import numpy

from pandas 		import DataFrame
from collections 	import deque


class Record:
	"""Class that handles timeseries records.

	:attr name: the name of the record.
	:type name: str.
	:attr maxlen: the maximum length of the record.
	:type maxlen: int.
	:attr t: the list of timestamps.
	:type t: deque.
	:attr x: the list of recorded values.
	:type x: deque.
	"""

	def __init__(self, name="Record", maxlen=None):
		"""Special method for class object construction.

		:param name: the name of the record (optional).
		:type name: str.
		:param maxlen: the maximum length of the record.
		:type maxlen: int.
		"""
		self.name = name
		self.maxlen = maxlen
		self.t = deque(maxlen=maxlen)
		self.x = deque(maxlen=maxlen)		
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {}
		_repr["name"] = self.name
		_repr["maxlen"] = self.maxlen
		_repr["t"] = self.t
		_repr["x"] = self.x
		return _repr

	def __str__(self):
		"""Special method for class object printable version.
		"""
		_str = []
		for key, item in self.__repr__().items():
			_str.append("{} = {}".format(key, item))
		return "{}({})".format(self.__class__.__name__, ", ".join(_str))

	def __len__(self):
		"""Special method for class object length.
		"""
		return len(self.t)

	def __getitem__(self, index):
		"""Special method for class object item accessibility.
		"""
		if index >= 0 and index < self.__len__():
			return (self.t[index], self.x[index])
		else:
			raise IndexError("record index out of range.")


	def append(self, a, b):
		"""Add a and b to the right side of the deques t and x.

		:param a: the timestamp.
		:type a: numpy datetime64.
		:param b: the value.
		:type b: any.
		"""
		self.t.append(a)
		self.x.append(b)
		return

	def appendleft(self, a, b):
		"""Add a and b to the left side of the deques t and x.

		:param a: the timestamp.
		:type a: numpy datetime64.
		:param b: the value.
		:type b: any.
		"""
		self.t.appendleft(a)
		self.x.appendleft(b)
		return

	def clear(self):
		"""Remove all elements from t and x,
		leaving them with length 0.
		"""
		self.t.clear()
		self.x.clear()
		return

	def extend(self, a, b):
		"""Extend the right side of the t and x deques
		by appending elements from the iterable argument.

		:param a: the timestamps.
		:type a: iterable.
		:param b: the values.
		:type b: iterable.
		"""
		self.t.extend(a)
		self.x.extend(b)
		return

	def extendleft(self, a, b):
		"""Extend the left side of the t and x deques
		by appending elements from the iterable argument.	

		:param a: the timestamps.
		:type a: iterable.
		:param b: the values.
		:type b: iterable.

		.. note:: 
			the series of left appends results 
			in reversing the order of elements 
			in the iterable argument.
		"""
		self.t.extendleft(a)
		self.x.extendleft(b)
		return

	def pop(self):
		"""Remove and return an element 
		from the right side of the deque.
		If no elements are present, raises an IndexError.
		"""
		t = self.t.pop()
		x = self.x.pop()
		return (t, x)

	def popleft(self):
		"""Remove and return an element 
		from the left side of the deque. 
		If no elements are present, raises an IndexError.
		"""
		t = self.t.popleft()
		x = self.x.popleft()
		return (t, x)

	def prune(self, p=None):
		"""Prunes the t and x deques.

		:param p: pruning parameter.
		:type p: int.
		"""
		if p is None:
			return (self.t, self.x)
		elif not isinstance(p, int):
			msg = "The pruning parameter must have type int."
			raise TypeError(msg)
		elif prune < 0:
			msg = "The pruning parameter must be a positive integer."
			raise ValueError(msg)
		else:
			return (self.t[::p], self.x[::p])

	def asnumpy(self, p=None):
		"""Returns the time and values as numpy arrays.

		:param p: pruning parameter (optional).
		:type p: int.

		:return: the time and val as arrays.
		:rtype: ndarray.
		"""
		(t, x) = self.prune(p)
		t = numpy.asarray(t)
		x = numpy.asarray(x)
		return (t, x)

	def aspandas(self, p=None):
		"""Returns the time and values as a pandas dataframe.

		:param prune: pruning parameter (optional).
		:type prune: int.

		:return: the time and the values.
		:rtype: pandas DataFrame.
		"""
		(t, x) = self.prune(p)
		df = DataFrame(data={"time":t, "x":x})
		return df.set_index("time")

	def mean(self):
		"""Returns the mean of the recorded values.

		:return: the average x value.
		:rtype: float.
		"""
		return float(numpy.mean(self.asnumpy()[1]))

	def std(self):
		"""Returns the standard deviation of the recorded values.

		:return: the average x value.
		:rtype: float.
		"""
		return float(numpy.std(self.asnumpy()[1]))