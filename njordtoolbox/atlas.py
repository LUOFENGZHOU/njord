#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import torch
from collections import deque
from collections import OrderedDict

# Import the indicators.
from . import indicators


class Atlas():
	"""Class that handles live dataset generation.

	:attr root: the data root.
	:type root: list<list<float>>.
	:attr col: the column names of the root.
	:type col: tuple(str).
	:attr lk: the look back horizon.
	:type lk: int.
	:attr norm: the normalisation methods.
	:type norm: list<dict>.
	:attr collen: the number of columns.
	:type collen: int.
	:attr maxlen: the maximum length of the root.
	:type maxlen: int.
	"""

	def __init__(self, col, lk, maxlen=1000):
		"""Special method for class object construction.

		:param col: the column names of the root.
		:type col: tuple(str)
		:param lk: the look-back horizon.
		:type lk: int.
		:param features: the column names of the root (optional).
		:type features: dict<str>.
		:param maxlen: the maximum length of the root (optional). 
		:type maxlen: int.
		"""
		self.root = deque(maxlen=maxlen)
		self.col = list(col)
		self.lk = lk
		self.norm = []		
		self.collen = len(col)
		self.maxlen = maxlen
		self.indicator = []
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		msg = []
		msg.append("")
		msg.append("# atlas object: ")
		for i, col in enumerate(self.col):
			msg.append("# col {:02n} ... = {}".format(i, col))
		msg.append("# lk ....... = {}".format(self.lk))
		msg.append("# norm ..... = {}".format(self.norm))
		msg.append("# collen ... = {}".format(self.collen))
		msg.append("# maxlen ... = {}".format(self.maxlen))
		msg.append("")
		return "\n".join(msg)

	def __str__(self):
		"""Special method for class object printable version.
		"""
		return self.__repr__()

	def __len__(self):
		"""Special method for class oject printable version.
		"""
		return self.root.__len__()

	def add_indicator(self, itype, n, marker=None):
		"""Add an indicator.
		"""
		indicator = getattr(indicators, itype)(n)
		labels = indicator.label(marker)
		for label in labels:
			self.col.append(label)
		self.indicator.append(indicator)		
		return

	def append(self, x):
		"""Add x to the right side of the root.

		:param x: data to append.
		:type x: list<float>.
		"""
		if len(x) != self.collen:
			msg = "inconsistent argument in append method."
			raise ValueError(msg)
		else:
			for indicator in self.indicator:
				x.append(indicator(x[3]))
			self.root.append(x)
		return

	def appendleft(self, x):
		"""Add x to the left side of the root.

		:param x: data to append.
		:type x: list<float>.
		"""
		self.root.appendleft(x)
		return

	def clear(self):
		"""Remove all elements from the root,
		leaving it with length 0.
		"""
		self.root.clear()
		for ind in self.indicator:
			ind.clear()
		return

	def extend(self, iterable):
		"""Extend the right side of the root 
		by appending elements from the iterable argument.
		"""
		self.root.extend(iterable)
		return

	def extendleft(self, iterable):
		"""Extend the left side of the root 
		by appending elements from the iterable argument.
		"""
		self.root.extendleft(iterable)
		return

	def pop(self):
		"""Remove and return an element 
		from the right side of the root.
		If no elements are present, raises an IndexError.		
		"""
		return self.root.pop()

	def popleft(self):
		"""Remove and return an element 
		from the left side of the deque.
		If no elements are present, raises an IndexError.		
		"""
		return self.root.popleft()

	def set_norm(self, marker, method, reference=None):
		"""Add features that have to be normalised
		"""
		norm = {"method":method}
		idx = []
		ref = []
		
		for i, col in enumerate(self.col):
			if marker in col:
				idx.append(i)

		for i, col in enumerate(self.col):
			if reference is None:
				if "#t" in col:
					ref.append(i)
			else:
				if reference == col:
					ref = [i]
					break

		norm["idx"] = idx
		norm["ref"] = ref
		if not idx:
			pass
		else:
			self.norm.append(norm)
		return

	def _normalise_normal(self, data, index, mean, std):
		data[:,index] = ( data[:,index] - mean ) / std
		return data

	def _normalise_pvt_change(self, data, index, ref):
		num = data[:,index]
		den = data[-1,ref]
		data[:,index] = 100.0 * ( num / ( den + 1.0E-8 ) - 1 )
		return data

	def _normalise_pct_change(self, data, index, ref):
		num = data[1:,index]
		den = data[0:-1,ref]
		if len(den.size()) == 1:
			den = den.reshape(-1, 1)
		data[1:,index] = 100.0 * ( num / ( den + 1.0E-8 ) - 1 )
		return data

	def _normalise_log_change(self, data, index, ref):
		num = data[1:,index]
		den = data[0:-1,ref]
		if len(den.size()) == 1:
			den = den.reshape(-1, 1)
		data[1:,index] = 100.0 * torch.log( num / ( den + 1.0E-8 ) )
		return data

	def normalise(self, data, norm):
		"""Normalise the data for the specified normalisation methods.

		:param data: the data to be normalised.
		:type data: numpy array.
		:param norm_method: the method name used for normalisation.
		:type norm_method: <str>.
		:param reference: the reference feature used for normalisation.
		:type reference: <str>.
		
		:return: the data.
		:rtype: numpy array.
		"""
		if norm["method"] == "pvt_change":
			data = self._normalise_pvt_change(data, norm["idx"], norm["ref"])
		elif norm["method"] == "pct_change":
			data = self._normalise_pct_change(data, norm["idx"], norm["ref"])
		elif norm["method"] == "log_change":
			data = self._normalise_log_change(data, norm["idx"], norm["ref"])
		else:
			pass
		return data

	def astorch(self, size):
		"""Returns the root as a torch tensor.
		"""
		try:
			return torch.Tensor(list(self.root)[-size:])
		except TypeError:
			return None

	def get(self):
		"""Process the current data sample.
		"""

		# Check the current length.
		if self.__len__() < self.lk + 1:
			return None

		# Get the current data as torch tensor.
		data = self.astorch(size=self.lk + 1)

		# Check if the data is valid.
		if data is None:
			return None

		# Run the normalisation method.
		for norm in self.norm:
			data = self.normalise(data, norm)
		data = data[-self.lk:,:]
		data = data.unsqueeze(0)
		return data

	def last(self):
		"""Returns the last element of the root.
		"""
		return OrderedDict(zip(self.col, self.root[-1]))

