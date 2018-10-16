#!/usr/bin/env python
# coding=utf-8


class EMA():
	"""Class that handles an exponential moving average.

	:attr n: the period of the ema.
	:type n: float.
	:attr alpha: weight decay.
	:type alpha: float.
	:attr ema: the current filtered value.
	:type ema: float.
	:attr count: the number of filterd values.
	:type count: int.
	"""

	def __init__(self, n):
		"""Special method for class construction.

		:param n: the ema period.
		:type n: float or int.
		"""
		if not isinstance(n, float) and not isinstance(n, int):
			raise TypeError("N is type {}".format(type(n)))

		self.n = float(n)
		self.alpha = 2.0 / ( self.n + 1.0 )
		self.ema = None
		self.count = 0
		return

	def __repr__(self):
		"""Sepcial method for class object representation.
		"""
		msg = {}
		msg.update({"n": self.n})
		msg.update({"alpha": self.alpha})
		msg.update({"ema": self.ema})
		msg.update({"count": self.count})
		return msg

	def __str__(self):
		"""Special method for class object printable version.
		"""
		msg = []
		for a, b in self.__repr__().items():
			msg.append("{} = {}".format(a, b))
		return "{}({})".format(self.__class__.__name__, ", ".join(msg))

	def __call__(self, x):
		"""Special method for class function-like call.

		:param x: the value to filter.
		:type x: float.

		:return: the filter value.
		:rtype: float.
		"""

		# Check the type of the value.
		if not isinstance(x, float):
			raise TypeError()
		
		# Append the value.
		self.append(x)

		# Return the value if count is consistent.
		if self.count > self.n:
			return self.ema
		else:
			return None

	def append(self, x):
		"""Append a new value x to the ema.

		:param x: the value to filter.
		:type x: float.
		"""
		if self.ema is not None:
			self.ema += self.alpha * ( x - self.ema )
		else:
			self.ema = x
		self.count += 1
		return

	def clear(self):
		"""Clear the ema from the memory.
		"""
		self.count = 0
		self.ema = None
		return

	def label(self, marker=None):
		"""Returns the indicator label.
		"""
		if marker is None:
			return ["{}_{}".format(self.__class__.__name__, int(self.n))]
		else:
			return ["{}_{}_{}".format(self.__class__.__name__, int(self.n), marker)]
