#!/usr/bin/env python
# coding=utf-8

import numpy
import pandas

from .record import Record
from .trades import Trades
from .wallet import Wallet


class Exchange():
	"""Class the handles trades realisation.

	:attr symbol: the asset pair name.
	:type symbol: str.
	:attr timestamp: the current time.
	:type timestamp: Datetime64.
	:attr ask_price: the current first ask_price.
	:type ask_price: list.
	:attr bid_price: the current first bid_price.
	:type bid_price: list.
	:attr ask_qty: the current first ask_qty.
	:type ask_qty: list.
	:attr bid_qty: the current first bid_qty.
	:type bid_qty: list.
	:attr trades: class that handles the trades history.
	:type trades: class.
	:attr wallet_base: class that handles the base wallet
	:type wallet_base: class.
	:attr wallet_quote: class that handles the quote wallet
	:type wallet_quote: class.
	:attr fees: the exchange fees.
	:type fees: float.	
	"""

	def __init__(self, platform, symbol, base_qty=0.0, quote_qty=0.0):
		"""Special method for class object construction.
		"""

		# Set the platform and symbol.
		self.platform = platform		
		self.symbol = symbol

		# Set the base wallet.
		base_asset = self.symbol[:len(self.symbol)//2]
		self.wallet_base = Wallet(base_asset, base_qty, name="base_wallet")

		# Set the quote wallet.
		quote_asset = self.symbol[len(self.symbol)//2:]
		self.wallet_quote = Wallet(quote_asset, quote_qty, name="quote_wallet")

		# Set the initial trading quantities.
		self.timestamp = None
		self.ask_price = []
		self.bid_price = []
		self.ask_qty = []
		self.bid_qty = []

		# Set the trades record.
		self.trades = Trades()

		# Set the wealth record.
		self.wealth = Record()

		# Update fees according to platform
		self.fees = self.update_fees()

		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		msg = []
		msg.append("# --- Exchange --- #")
		msg.append("# symbol .... = {}".format(self.symbol))
		msg.append("# fees ...... = {}".format(self.fees))
		msg.append("# -------------- #")
		return "\n".join(msg)

	def __str__(self):
		"""Special method for class object printable version.
		"""
		return self.__repr__()

	def clear(self):
		"""Clear the exchange from the memory.
		"""
		self.trades.clear()
		self.wealth.clear()
		return

	def update(self, t, ask_p, ask_q, bid_p, bid_q):
		"""Update current timestamp, prices and quantities.
		
		:param t: the current timestamp.
		:type t: numpy datetime64.
		:param ask_p: the current first ask order prices.
		:type ask_p: list.
		:param ask_q: the current first ask order qties.
		:type ask_q: list.		
		:param bid_p: the current first bid order prices.
		:type bid_p: list.
		:param bid_q: the current first bid order qties.
		:type bid_q: list.		
		"""
		self.timestamp = t
		self.ask_price = ask_p
		self.ask_qty = ask_q		
		self.bid_price = bid_p
		self.bid_qty = bid_q
		return

	def check_bid_qty(self, base_qty):
		"""Check the bid side of the orderbook for available qty
		
		:attr base_qty: the base qty to check.
		:type base_qty: float.
		"""
		initial_qty = base_qty
		final_qty = 0.0
		trade = 0.0
		fees = 0.0
		gain = 0.0

		for q, p in zip(self.bid_qty, self.bid_price):

			if q > base_qty:
				trade += p*base_qty
				fees += (p*base_qty) * self.fees
				gain = trade - fees
				base_qty = 0.0
				break

			else:
				trade += p*q
				fees += (p*q) * self.fees
				gain = trade - fees
				base_qty -= q
		
		if base_qty == 0:
			final_qty = initial_qty
		else:
			final_qty = initial_qty - base_qty	
	
		return gain, fees, final_qty, p

	def check_ask_qty(self, base_qty):
		"""Check the ask side of the orderbook for available qty
		
		:attr base_qty: the base qty to check.
		:type base_qty: float.
		"""
		initial_qty = base_qty
		final_qty = 0.0
		trade = 0.0
		fees = 0.0
		cost = 0.0

		for q, p in zip(self.ask_qty, self.ask_price):

			if q > base_qty:
				trade += p * base_qty
				fees += (p * base_qty) * self.fees
				cost = trade + fees
				base_qty = 0.0
				if self.wallet_quote.qty - cost >= 0.0:
					break
				else:
					trade = 0.0
					fees = 0.0
					cost = 0.0
					base_qty = initial_qty
					break

			else:
				trade += p*q
				fees += (p*q) * self.fees
				cost = trade + fees
				base_qty -= q		
				
		if base_qty == 0:
			final_qty = initial_qty
		elif base_qty == initial_qty:
			final_qty = 0
		else:
			final_qty = initial_qty - base_qty

		return cost, fees, final_qty, p

	def buy_market(self, base_qty):
		"""Realise a buy market trade.

		:attr base_qty: the base qty to buy.
		:type base_qty: float.
		"""

		cost, fees, buy_qty, buy_price = self.check_ask_qty(base_qty)

		if cost <= self.wallet_quote.qty and cost > 0:
			self.wallet_base.plus(buy_qty)
			self.wallet_quote.minus(cost)
			self.trades.append(self.timestamp, buy_price, 
				(cost-fees), buy_qty, fees, "buy")
		if buy_qty < base_qty:
			print("Buy market: Market liquidities are not sufficient to realise full trade.")
		elif cost > self.wallet_quote.qty:
			print("Buy market: Your funds are not sufficient to realise this trade.")

		return


	def sell_market(self, base_qty):
		"""Realise a sell market trade.
		
		:param base_qty: the base qty to sell.
		:type base_qty: float.
		"""

		gain, fees, sell_qty, sell_price = self.check_bid_qty(base_qty)

		if sell_qty <= self.wallet_base.qty and sell_qty > 0:
			self.wallet_base.minus(sell_qty)
			self.wallet_quote.plus(gain)
			self.trades.append(self.timestamp, sell_price, 
				(gain-fees), sell_qty, fees, "sell")
		if sell_qty < base_qty:
			print("Sell market: Market liquidities are not sufficient to realise full trade.")
		
		return

	def buy_limit(self, base_qty, premium):
		"""Realise a buy limit order.
		
		:param base_qty: the base qty to buy.
		:type base_qty: float.
		:param premium: the premium of the spread to add to the first bid price
		:type premium: float.
		"""
		spread = self.ask_price[0] - self.bid_price[0]
		bid_price = self.bid_price[0] + ( spread * premium )
		quote_qty = base_qty * bid_price
		fees_qty = quote_qty * self.fees
		cost = quote_qty + fees_qty

		if cost <= self.wallet_quote.qty:
			self.wallet_base.plus(base_qty)
			self.wallet_quote.minus(cost)
			self.trades.append(self.timestamp, bid_price, 
				quote_qty, base_qty, fees_qty, "buy")
		else:
			print("Buy limit: Your funds are not sufficient to realise this trade")

		return "buy", bid_price

	def sell_limit(self, base_qty, premium):
		"""Realise a sell limit order
		
		:param base_qty: the base qty to sell.
		:type base_qty: float.
		:param premium: the premium of the spread to subtract to the first bid price
		:type premium: float.
		"""
		spread = self.ask_price[0]-self.bid_price[0]
		ask_price = self.ask_price[0] - (spread*premium)
		quote_qty = base_qty * ask_price
		fees_qty = quote_qty * self.fees
		gain = quote_qty - fees_qty

		if base_qty <= self.wallet_base.qty:
			self.wallet_base.minus(base_qty)
			self.wallet_quote.plus(gain)
			self.trades.append(self.timestamp, ask_price, 
				quote_qty, base_qty, fees_qty, "sell")
		else:
			print("Sell limit: Your funds are not sufficient to realise this trade")

		return "sell", ask_price

	def buy(self, ordertype, base_qty, premium=0.001):
		"""Realise a buy order.

		:param ordertype: the order type.
		:type ordertype: str.
		:param base_qty: the base quantity to buy.
		:type base_qty: float.
		:param premium: the premium to add to the price to be the first order.
		:type premium: float.
		"""
		if ordertype == "market":
			self.buy_market(base_qty)
		elif ordertype == "limit":
			self.buy_limit(base_qty, premium)
		else:
			raise ValueError("Please provide a consistent ordertype.")
		return

	def sell(self, ordertype, base_qty, premium=0.001):
		"""Realise a buy order.

		:param ordertype: the order type.
		:type ordertype: str.
		:param base_qty: the base quantity to sell.
		:type base_qty: float.
		:param premium: the premium to substract to the price to be the first order.
		:type premium: float.		
		"""
		if ordertype == "market":
			self.sell_market(base_qty)
		elif ordertype == "limit":
			self.sell_limit(base_qty, premium)
		else:
			raise ValueError("Please provide a consistent ordertype.")
		return

	def get_trades_buy(self):
		"""Returns the buy orders trades.

		:return timestamp: the order timestamps.
		:rtype timestamp: numpy array.
		:return order_price: the order prices.
		:rtype order_price: numpy array.
		"""
		trades = self.trades.aspandas()
		buy = trades[trades["o"] == "buy"]
		timestamp = buy.index.values
		order_price = buy.loc[:,"p"].values
		return (timestamp, order_price)

	def get_trades_sell(self):
		"""Returns the sell orders trades.

		:return timestamp: the order timestamps.
		:rtype timestamp: numpy array.
		:return order_price: the order prices.
		:rtype order_price: numpy array.		
		"""
		trades = self.trades.aspandas()
		buy = trades[trades["o"] == "sell"]
		timestamp = buy.index.values
		order_price = buy.loc[:,"p"].values
		return (timestamp, order_price)

	def get_trades(self):
		"""Returns the trade informations.

		:return order_price: the trade informations.
		:rtype order_price: pandas dataframe.		
		"""
		return self.trades.aspandas()

	def get_trades_sum(self):
		"""Returns the quote sum of the trades.

		:return trades: the sum of the trade in quote currency
		:rtype trades: float.		
		"""
		trades = self.trades.aspandas()
		return numpy.sum(trades.loc[:,"b"].values)

	def get_fees_sum(self):
		"""Returns the quote sum of the trades fees.

		:return fees: the sum of the trade fees in quote currency
		:rtype fees: float.			
		"""
		trades = self.trades.aspandas()
		return numpy.sum(trades.loc[:,"f"].values)

	def get_wealth(self):
		"""Returns the actual wealth.

		:return wealth: the wealth in quote currency
		:rtype wealth: float.			
		"""
		wealth = self.wealth.aspandas()
		return wealth.iloc[-1,-1]

	def update_fees(self):
		"""Update the fees.
		"""
		total_trades = self.get_trades_sum()

		if self.platform == "kraken":
			if total_trades > 10.0E+6:
				fees = 0.00
			elif total_trades < 0.050E+6:
				fees = 0.16
			elif total_trades < 0.100E+6:
				fees = 0.14
			elif total_trades < 0.250E+6:
				fees = 0.12
			elif total_trades < 0.500E+6:
				fees = 0.10
			elif total_trades < 1.000E+6:
				fees = 0.08
			elif total_trades < 2.500E+6:
				fees = 0.06
			elif total_trades < 5.000E+6:
				fees = 0.04
			else:
				fees = 0.02

		elif self.platform == "bitfinex":
			if total_trades > 7.5E+6:
				fees = 0.00
			elif total_trades < 0.050E+6:
				fees = 0.1
			elif total_trades < 0.500E+6:
				fees = 0.08
			elif total_trades < 1.000E+6:
				fees = 0.06
			elif total_trades < 2.500E+6:
				fees = 0.04
			else:
				fees = 0.02

		elif self.platform == "binance":
			fees = 0.1
		else:
			raise Exception
			
		return fees / 100

	def update_wealth(self):
		"""Update the wealth.
		"""
		if isinstance(self.bid_price, list):
			try:
				wealth = self.wallet_quote.qty + self.wallet_base.qty * self.bid_price[0]
			except Exception as msg:
				print(self.bid_price)
				raise ValueError()
			self.wealth.append(self.timestamp, wealth)
		else:
			pass
		return

	def buy_dummy_market(self, base_qty):
		"""Process a dummy buy order.
		
		:param base_qty: the quote qty to buy.
		:type base_qty: float.
		"""

		# Get the current ask price
		price = self.ask_price[0]

		# Compute the quote quantity to sell.
		quote_qty = base_qty * price

		# Compute the fees.
		fees = quote_qty * self.fees

		# Compute the final quote quantity.
		quote_qty = quote_qty + fees

		if self.wallet_quote.qty < quote_qty:
			pass #print("Buy dummy market: not enough quote funds.")
		else:
			self.wallet_quote.minus(quote_qty)
			self.wallet_base.plus(base_qty)
			self.trades.append(self.timestamp, 
				price, base_qty, quote_qty, fees, "buy")

		return

	def sell_dummy_market(self, base_qty):
		"""Process a dummy sell order.
		
		:param base_qty: the base quantity to sell.
		:type base_qty: float.
		"""
		if self.wallet_base.qty < base_qty:
			#print("Buy dummy market: not enough base funds.")
			return

		# Get the current bid price.
		price = self.bid_price[0]

		# Compute the quote quantity to buy.
		quote_qty = base_qty * price

		# Compute the fees.
		fees = quote_qty * self.fees
		
		# Compute the final quote quantity.
		quote_qty = quote_qty - fees

		if self.wallet_base.qty < base_qty:
			pass#print("Sell dummy market: not enough base funds.")
		else:
			self.wallet_quote.plus(quote_qty)
			self.wallet_base.minus(base_qty)
			self.trades.append(self.timestamp, 
				price, base_qty, quote_qty, fees, "sell")

		return
