#!/usr/bin/env python
# coding=utf-8

import numpy
import torch
from collections import deque, OrderedDict


class Window():
	"""Class that handles a rolling window data generation.


	:attr columns: the name of the columns.
	:type columns: list<str> or tuple(str).
	:attr lookback: the sliding window length.
	:type lookback: int.
	:attr collen: the number of columns.
	:type collen: int.
	:attr maxlen: the maximum length of the root.
	:type maxlen: int.
	:attr root: the data root for the sliding widnow.
	:type root: deque(list)
	:attr norm: the normalisation methods.
	:type norm: dict.
	"""

	def __init__(self, lookback, maxlen=None):
		"""Special method for class object construction.

		:param lookback: the look-back horizon.
		:type lookback: int.
		:param maxlen: the maximum length of the root (optional). 
		:type maxlen: int.
		"""
		self.collen = None
		self.columns = None
		self.lookback = lookback
		if maxlen is None:
			self.maxlen = lookback+10
		else:
			self.maxlen = maxlen
		self.root = deque(maxlen=self.maxlen)
		self.norm = []
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {
			"columns": self.columns,
			"lookback": self.lookback,
			"collen": self.collen,
			"maxlen": self.maxlen,
			"root": self.root,
			"norm": self.norm}
		return _repr

	def __str__(self):
		"""Special method for class object printable version.
		"""
		_str = []
		for key, item in self.__repr__().items():
			_str.append("{} = {}".format(key, item))
		return "{}({})".format(self.__class__.__name__, ", ".join(_str))

	def __len__(self):
		"""Special method for class oject printable version.
		"""
		return self.root.__len__()

	def __call__(self):
		"""Special method for class object function-like call.
		"""

		# Check the length of the current root.
		if self.__len__() <= self.lookback:
			return None

		# Check the user specified return data type.
		data = self.astorch()

		# Normalise the data.
		for norm in self.norm:
			data = self.normalise(data, norm)

		# Update the data.
		data = data[-self.lookback:,:]
		data = data.unsqueeze(0)

		# Returns the data.
		return data

	# ---------------------------- #
	# --- Window basic methods --- #
	# ---------------------------- #

	def append(self, x):
		"""Add x to the right side of the root.

		:param x: data to append.
		:type x: dict.
		"""

		# Check the type of x.
		if not isinstance(x, dict):
			raise TypeError("Please provide x a dict.")

		# Add columns if required.
		if self.columns is None:
			self.columns = list(x.keys())
			self.collen = len(self.columns)
			for i, norm in enumerate(self.norm):
				self.norm[i] = self.set_idx_norm(norm)
		else:
			pass

		# Append values to root.
		self.root.append(list(x.values()))

		return

	def clear(self):
		"""Remove all elements from the root,
		leaving it with length 0.
		"""
		self.root.clear()
		return

	def extend(self, iterable):
		"""Extend the right side of the root 
		by appending elements from the iterable argument.
		"""
		for item in iterable:
			self.root.extend(item)
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

	# --------------------------- #
	# --- Window main methods --- #
	# --------------------------- #

	def last(self):
		"""Returns the last element of the root.
		"""
		return OrderedDict(zip(self.columns, self.root[-1]))

	def astorch(self):
		"""Returns the root as a torch tensor.
		"""
		try:
			return torch.Tensor(list(self.root)[-self.lookback-1:])
		except TypeError:
			print("TypeError in astorch method.")
			return None

	def asnumpy(self):
		"""Returns the root as a numpy array.
		"""
		try:
			return numpy.asarray(list(self.root)[-self.lookback-1:])
		except TypeError:
			print("TypeError in astorch method.")
			return

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

	# ------------------------------------ #
	# --- Window normalisation methods --- #
	# ------------------------------------ #

	def add_norm(self, marker, method, ref=None):
		"""Add features that have to be normalised

		:param marker: the marker name associated to the column.
		:type marker: str.
		:param method: the name of the normalisation method.
		:type method: str.
		:param ref: reference for normalisation.
		:type ref: str.
		"""

		# Set the normalisation method as a dictionnary.
		norm = {
			"marker": marker, 
			"method": method,
			"ref": ref,
			"idx": None
		}

		# Set the normalisation index and reference.
		if self.columns is not None:
			norm = self.set_index_norm(norm)

		# Append the normalisation method.
		self.norm.append(norm)

		return

	def set_idx_norm(self, norm):
		"""Returns the normalisation method with updated index and reference.

		:param norm: the normalisation method.
		:type norm: dict.

		:return: the updated normalisation method.
		:rtype: dict.
		"""

		# Assign the indices.
		idx = []		
		for i, column in enumerate(self.columns):
			if norm["marker"] in column:
				idx.append(i)

		# Assign the reference.
		ref = []
		for i, column in enumerate(self.columns):
			if norm["ref"] is None:
				if norm["marker"] in column:
					ref.append(i)
			else:
				if norm["ref"] == column:
					ref = [i]
					break

		# Update and return the norm.
		norm["idx"] = idx
		norm["ref"] = ref

		return norm

	def normalise(self, data, norm):
		"""Normalise the data for the specified normalisation methods.

		:param data: the data to be normalised.
		:type data: numpy array or torch tensor.
		:param norm_method: the normalisation method.
		:type norm_method: dict.
		
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
			raise ValueError("Unsupported method {}".format(norm["method"]))
		return data

	def _normalise_normal(self, data, index, mean, std):
		data[:,index] = ( data[:,index] - mean ) / ( std + 1.0E-8 )
		return data

	def _normalise_pvt_change(self, data, index, ref):
		num = data[:,index]
		den = data[-1,ref]
		data[:,index] = 100.0 * ( num / ( den + 1.0E-12) - 1 )
		return data

	def _normalise_pct_change(self, data, index, ref):
		num = data[1:,index]
		den = data[0:-1,ref]
		if len(den.size()) == 1:
			den = den.reshape(-1, 1)
		data[1:,index] = 100.0 * ( num / ( den + 1.0E-12 ) - 1 )
		return data

	def _normalise_log_change(self, data, index, ref):
		num = data[1:,index]
		den = data[0:-1,ref]
		if len(den.size()) == 1:
			den = den.reshape(-1, 1)
		data[1:,index] = 100.0 * torch.log( num / ( den + 1.0E-12 ) )
		return data
