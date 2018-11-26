#!/usr/bin/env python
# coding=utf-8

from .ema import EMA


class RSI():
	"""Class that handles the rsi compute.

	:attr close: the previous close price.
	:type close: float.
	:attr u_ema: the upwward trend ema.
	:type u_ema: EMA object.
	:attr d_ema: the downward trend ema.
	:type d_ema: EMA object.	
	"""

	def __init__(self, n=9):
		"""Special method for class construction.

		:param n: the rsi period.
		:type n: int.
		"""
		self.n = n
		self.close = None
		self.u_ema = EMA(n)
		self.d_ema = EMA(n)
		self.count = 0
		self.rsi = None
		return

	def __call__(self, x):
		"""Special method for class function-like call.

		:param x: the new close price.
		:type x: float.		
		"""

		# Check the type of x.
		if not isinstance(x, float):
			raise ValueError()

		# Append x to the rsi.
		self.append(x)

		# Return the right value for the rsi.
		if self.count > self.n:
			return self.rsi
		else:
			return None

	def clear(self):
		"""Clear the ema from the memory.
		"""
		self.count = 0
		self.close = None
		self.u_ema.clear()
		self.d_ema.clear()
		return

	def append(self, x):
		"""Append a new value x to the ema.

		:param x: the new close price.
		:type x: float.
		"""
		if self.close is not None:

			# Compute the upward change and filter it.
			u = max(0.0, x - self.close)
			u_ema = self.u_ema(u)

			# Compute the downward change and filter it.
			d = max(0.0, self.close - x)
			d_ema = self.d_ema(d)

			# Compute the rsi.
			if u_ema is not None and d_ema is not None:
				self.rsi = 1.0 - 1.0 / ( 1.0 + u_ema / d_ema )
			else:
				self.rsi = None

		else:
			self.rsi = None	

		# Move to the next close price.
		self.close = x

		# Increment the counter.
		self.count += 1

		return
