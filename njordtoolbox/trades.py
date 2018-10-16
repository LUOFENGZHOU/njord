#!/usr/bin/env python
# coding=utf-8

from numpy			import asarray
from pandas			import DataFrame
from collections 	import deque


class Trades():
	"""Class that handles trades records.

	:attr symbol: the asset pair name.
	:type symbol: str.
	:attr t: the trade timestamps.
	:type t: deque<datetime64>.
	:attr p: the trade prices.
	:type p: deque<float>.
	:attr b: the trade quote qty.
	:type b: deque<float>.
	:attr q: the trade base qty.
	:type q: deque<float>.
	:attr f: the trade fees in the quote currency.
	:type f: deque<float>.
	:attr o: order type, buy or sell.
	:type o: deque<str>.
	"""


	def __init__(self, name="trade", maxlen=1000000):
		"""Special method for class object construction.

		:param name: the name of the asset pair.
		:type name: str.
		"""
		self.name = name
		self.maxlen = maxlen
		self.t = deque(maxlen=maxlen)
		self.p = deque(maxlen=maxlen)
		self.b = deque(maxlen=maxlen)
		self.q = deque(maxlen=maxlen)
		self.f = deque(maxlen=maxlen)
		self.o = deque(maxlen=maxlen)
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		return ", ".join([self.name, str(self.__len__())])

	def __str__(self):
		"""Special method for class object printable version.
		"""
		return self.__repr__()

	def __len__(self):
		"""Special method for class object length.
		"""
		return self.t.__len__()

	def __getitem__(self, index):
		"""Special method for class object item accessibility.
		"""
		if index >= 0 and index < self.__len__():
			t = self.t[index]
			p = self.p[index]
			b = self.b[index]
			q = self.q[index]
			f = self.f[index]
			o = self.o[index]
			return (t, p, b, q, f, o)
		else:
			raise IndexError("record index out of range.")


	def append(self, t, p, b, q, f, o):
		"""Add a trade to the right side of the trade deques.

		:param t: the timestamp.
		:type t: numpy datetime64.
		:param p: the price.
		:type p: float.
		:param b: the quote qty.
		:type b: float.
		:param q: the base qty.
		:type q: float.
		:param f: the fees.
		:type f: float.
		:param o: transaction type, buy or sell.
		:type o: str.
		"""
		self.t.append(t)
		self.p.append(p)
		self.b.append(b)
		self.q.append(q)
		self.f.append(f)
		self.o.append(o)
		return

	def appendleft(self, t, p, b, q, f, o):
		"""Add a trade to the left side of the trade deques.

		:param t: the timestamp.
		:type t: numpy datetime64.
		:param p: the price.
		:type p: float.
		:param b: the quote qty.
		:type b: float.
		:param q: the base qty.
		:type q: float.
		:param f: the fees.
		:type f: float.
		:param o: transaction type, buy or sell.
		:type o: str.
		"""
		self.t.appendleft(t)
		self.p.appendleft(p)
		self.b.appendleft(b)
		self.q.appendleft(q)
		self.f.appendleft(f)
		self.o.appendleft(o)
		return

	def clear(self):
		"""Remove all elements from the trade deques,
		leaving it with length 0.
		"""
		self.t.clear()
		self.p.clear()
		self.b.clear()
		self.q.clear()
		self.f.clear()
		self.o.clear()
		return

	def extend(self, t, p, b, q, f, o):
		"""Extend the right side of the trade deques
		by appending elements from the iterable arguments.

		:param t: the timestamps.
		:type t: iterable<numpy datetime64>.
		:param p: the prices.
		:type p: iterable<float>.
		:param b: the quote qties.
		:type b: iterable<float>.
		:param q: the base qties.
		:type q: iterable<float>.
		:param f: the fees.
		:type f: iterable<float>.
		:param o: transaction type, buy or sell.
		:type o: iterable<str>.
		"""
		self.t.extend(t)
		self.p.extend(p)
		self.b.extend(b)
		self.q.extend(q)
		self.f.extend(f)
		self.o.extend(o)
		return

	def extendleft(self, t, p, b, q, f, o):
		"""Extend the left side of the trade deques
		by appending elements from the iterable arguments.

		:param t: the timestamps.
		:type t: iterable<numpy datetime64>.
		:param p: the prices.
		:type p: iterable<float>.
		:param b: the quote qties.
		:type b: iterable<float>.
		:param q: the base qties.
		:type q: iterable<float>.
		:param f: the fees.
		:type f: iterable<float>.
		:param o: transaction type, buy or sell.
		:type o: iterable<str>.

		.. note::
			the series of left appends results 
			in reversing the order of elements 
			in the iterable arguments.
		"""
		self.t.extendleft(t)
		self.p.extendleft(p)
		self.b.extendleft(b)
		self.q.extendleft(q)
		self.f.extendleft(f)
		self.o.extendleft(o)
		return

	def pop(self):
		"""Remove and return an element 
		from the right side of the trade deques. 
		If no elements are present, raises an IndexError.
		"""
		t = self.t.pop()
		p = self.p.pop()
		b = self.b.pop()
		q = self.q.pop()
		f = self.f.pop()
		o = self.o.pop()
		return (t, p, b, q, f, o)

	def popleft(self):
		"""Remove and return an element
		from the left side of the trade deques. 
		If no elements are present, raises an IndexError.
		"""
		t = self.t.popleft()
		p = self.p.popleft()
		b = self.b.popleft()
		q = self.q.popleft()
		f = self.f.popleft()
		o = self.o.popleft()
		return (t, p, b, q, f, o)		

	def asnumpy(self):
		"""Returns the trade deques as numpy arrays.
		"""
		t = asarray(self.t)
		p = asarray(self.p)
		b = asarray(self.b)
		q = asarray(self.q)
		f = asarray(self.f)
		o = asarray(self.o)
		return (t, p, b, q, f, o)

	def aspandas(self):
		"""Returns the trade deques as pandas dataframes.
		"""
		data = {}
		data["time"] = self.t
		data["p"] = self.p
		data["b"] = self.b
		data["q"] = self.q
		data["f"] = self.f
		data["o"] = self.o
		df = DataFrame(data=data)
		return df.set_index("time")

	def sum_base(self):
		"""Computes the total traded base volume.
		"""
		return sum(self.b)

	def sum_quote(self):
		"""Computes the total traded quoted volume.
		"""
		return sum(self.q)

	def sum_fees(self):
		"""Computes the total fees.
		"""
		return sum(self.f)

	def get_buy_orders(self):
		"""Get the buy orders.
		"""
		trades = self.aspandas()
		buy = trades[trades["order"] == "buy"]
		t = buy.index.values
		p = buy.loc[:,"p"].values
		return (t, p)

	def get_sell_orders(self):
		"""Get the sell orders"
		"""
		trades = self.aspandas()
		sell = trades[trades["order"] == "sell"]
		t = sell.index.values
		p = sell.loc[:,"p"].values
		return (t, p)
