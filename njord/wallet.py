#!/usr/bin/env python
# coding=utf-8


class Wallet():
	"""Class that handles a wallet.

	:attr qty: the wallet volume.
	:type qty: float.	
	:attr asset: the asset name.
	:type asset: str.
	:attr name: the name of the wallet.
	:type name: str.
	"""

	def __init__(self, asset, qty=0.0, name="wallet"):
		"""Special method for class object construction.

		:param asset: the name of the asset.
		:type asset: str.
		:param qty: the wallet initial volume (optional).
		:type qty: float.
		:param name: the name of the wallet (optional).
		:type name: str.
		"""
		assert(isinstance(qty, float))		
		assert(isinstance(name, str))
		assert(isinstance(asset, str))		
		
		self.qty = qty
		self.name = name		
		self.asset = asset
		self._qty_start = qty			
		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		msg = []
		msg.append("# {}: #".format(self.name))
		msg.append("# asset ... = {}".format(self.asset))
		msg.append("# qty ..... = {}".format(self.qty))
		return "\n".join(msg)

	def __str__(self):
		"""Special method for class object printable version.
		"""
		return self.__repr__()

	def clear(self):
		"""Reset the wallet.
		"""
		self.qty = self.qty_initial
		return 

	def minus(self, qty):
		"""Sell qty from the wallet.
		"""
		self.qty -= qty
		return qty

	def plus(self, qty):
		"""Buy qty and add to wallet.
		"""
		self.qty += qty
		return True




