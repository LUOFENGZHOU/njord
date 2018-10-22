#!/usr/bin/env python
# coding=utf-8

from numpy			import asarray
from pandas			import DataFrame
from collections 	import deque, namedtuple


class Trades():
	"""Class that handles trades records.

	:attr name: the name of the trade records.
	:type name: str.
	:attr maxlen: the maximum length of the trades.
	:type maxlen: int.
	:attr trades: the list of recorded trades.
	:type trades: deque<namedtuple>
	"""

	Trade = namedtuple("Trade", 
		["time", "price", "base", "quote", "fees", "type"])


	def __init__(self, name="trade", maxlen=None):
		"""Special method for class object construction.

		:param name: the name of the asset pair.
		:type name: str.
		"""
		self.name = name
		self.maxlen = maxlen
		self.trades = deque(maxlen=maxlen)
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {
			"name": self.name,
			"maxlen": self.maxlen,
			"trade": self.trades,
			}
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
		return self.trades.__len__()

	def append(self, t, price, base, quote, fees, otype):
		"""Add a trade to the right side of the trade deques.

		:param t: the timestamp.
		:type t: numpy datetime64.
		:param price: the order base price.
		:type price: float.
		:param base: the base qty.
		:type base: float.
		:param quote: the base qty.
		:type quote: float.
		:param fees: the trade fees in base currency.
		:type fees: float.
		:param otype: the transaction type, 'buy' or 'sell'.
		:type otype: str.
		"""
		self.trades.append(self.Trade(t, price, base, quote, fees, otype))
		return

	def appendleft(self, t, p, b, q, f, o):
		"""Add a trade to the left side of the trade deques.

		:param t: the timestamp.
		:type t: numpy datetime64.
		:param price: the order base price.
		:type price: float.
		:param base: the base qty.
		:type base: float.
		:param quote: the base qty.
		:type quote: float.
		:param fees: the trade fees in base currency.
		:type fees: float.
		:param otype: the transaction type, 'buy' or 'sell'.
		:type otype: str.
		"""
		self.trades.appendleft(self.Trade(t, price, base, quote, fees, otype))
		return

	def clear(self):
		"""Remove all elements from the trade deques,
		leaving it with length 0.
		"""
		self.trades.clear()
		return

	def pop(self):
		"""Remove and return an element 
		from the right side of the trade deques. 
		If no elements are present, raises an IndexError.
		"""
		x = self.trades.pop()
		return (x.time, x.price, x.base, x.quote, x.fees, x.type)

	def popleft(self):
		"""Remove and return an element
		from the left side of the trade deques. 
		If no elements are present, raises an IndexError.
		"""
		x = self.trades.popleft()
		return (x.time, x.price, x.base, x.quote, x.fees, x.type)	

	def astuple(self):
		"""Returns the trade deques as tuple.
		"""
		return self.Trade(*zip(*self.trades))

	def aspandas(self):
		"""Returns the trades.
		"""		
		df = DataFrame(list(self.trades))
		return df.set_index("time")

	def buy_orders(self):
		"""Returns the buy orders.
		"""
		trades = self.aspandas()
		return trades[trades["type"] == "buy"]

	def sell_orders(self):
		"""Get the sell orders.
		"""
		trades = self.aspandas()
		return trades[trades["type"] == "sell"]
