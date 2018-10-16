#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import ast
import numpy

# Import njordtoolbox packages.
from .record 	import Record
from .history 	import History
from .exchange  import Exchange


class Environment:
	"""Class that handles a trading environment for RL.

	:attr platform: the name of the trading platform.
	:type platform: str.
	:attr dataset: the name of the dataset.
	:type dataset: str.
	:attr symbol: the name of the symbol.
	:type symbol: str.
	:attr period: the trading period.
	:type period: int.
	:attr version: the dataset version.
	:type version: str.
	"""

	label_ask_p = "close_ask_price_1_10"
	label_ask_q = "close_ask_qty_1_10"
	label_bid_p = "close_bid_price_1_10"
	label_bid_q = "close_bid_qty_1_10"

	def __init__(self, platform, symbol, public, market, period=None):
		"""Special method for class initialisation.

		:param platform: the name of the trading platform.
		:type platform: str.
		:param dataset: the name of the dataset.
		:type dataset: str.
		:param symbol: the name of the symbol.
		:type symbol: str.
		:param period: the trading period.
		:type period: int.
		"""

		# Set the attributes.
		self.platform = platform
		self.symbol = symbol

		# Set the period.
		self.period = period

		# Load the public history.
		self.public = self._load_history_public(public)
		self.public.scope("close_price")

		# Load the market history.
		self.market = self._load_history_market(market)
		self.market.scope("ask_price")
		self.market.scope("bid_price")

		# Set the trade quantity.
		self.trade_qty = 0.1

		# Set the exchange.
		self.exchange = Exchange(self.platform, self.symbol, quote_qty=0.0)

		# Set the records.
		self.market_pnl = Record("market_pnl")
		self.realized_pnl = Record("realized_pnl")
		self.realized_pnl_pct = Record("realized_pnl_pct")
		self.unrealized_pnl = Record("unrealized_pnl")
		self.unrealized_pnl_pct = Record("unrealized_pnl_pct")
		self.actions = Record("actions")
		self.rewards = Record("rewards")
		self.returns = Record("returns")

		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		msg = {}
		msg.update({"platform": self.platform})
		msg.update({"symbol": self.symbol})
		msg.update({"period": self.period})
		return msg

	def __str__(self):
		"""Special method for class object printable version.
		"""
		msg = []
		for a, b in self.__repr__().items():
			msg.append("{} = {}".format(a, b))
		return "{}({})".format(self.__class__.__name__, ", ".join(msg))

	def _load_history_public(self, public):
		public = public.dropna()
		public = public.asfreq("{}S".format(self.period), method="ffill")
		return History(public, "Public")

	def _load_history_market(self, market):
		market = market.dropna()
		market = market.asfreq("{}S".format(self.period), method="ffill")
		for col in market.columns:
			market.loc[:,col] = market.loc[:,col].apply(
				lambda x: ast.literal_eval(x))
		return History(market, "Market")		

	# ------------------------------------ #
	# --- 0. Environment basic methods --- #
	# ------------------------------------ #

	def clear(self):
		"""Clear the environment from the memory.
		"""
		self.market.clear()
		self.public.clear()
		self.exchange.clear()
		self.actions.clear()
		self.rewards.clear()
		return

	def reset(self, quote_qty=0.0, random=True):
		"""Reset the exchange.

		:param quote_qty: the initial quote qty.
		:type quote_qty: float.
		:param random: If true the initial timestamp is taken randomly.
		:type random: bool.
		"""

		# Reset the public history timestamp.
		if random is True:
			self.public.timestamp = self.public.randomtimestamp()
		else:
			self.public.timestamp = self.public.timestamp_first

		# Reset the market history timestamp.
		self.market.timestamp = self.public.timestamp

		# Set a new exchange.
		self.exchange = Exchange(platform=self.platform, symbol=self.symbol, 
			quote_qty=quote_qty)

		# Update the exchange.
		self.update_exchange()
		
		# Set the initial recusion values.
		# Update the current ask and bid price.
		self._wstart = self.exchange.get_wealth()
		self._wealth = self.exchange.get_wealth()
		self._action = 0
		self.buy_position_price = None
		self.buy_position_wealth = None
		self.sell_position_price = None
		self.sell_position_wealth = None
		self.market_position = self.exchange.ask_price[0]

		self.market_pnl.clear()
		self.realized_pnl.clear()
		self.realized_pnl_pct.clear()
		self.unrealized_pnl.clear()
		self.unrealized_pnl_pct.clear()

		return		

	def update_exchange(self):
		"""Update the exchange with current market conditions.
		"""

		# Extract the current market conditions.
		market = self.market.get_dict_values()

		# Update the exchange with the market conditions.
		self.exchange.update(self.market.timestamp, 
			market[self.label_ask_p], market[self.label_ask_q], 
			market[self.label_bid_p], market[self.label_bid_q])

		# Update the client wealth portfolio.
		self.exchange.update_wealth()

		# Update the current wealth
		self._wealth = self.exchange.get_wealth()

		return

	def update_market_pnl(self):
		"""Update the market pnl.
		"""

		# Current timestamp.
		timestamp = self.market.timestamp		

		# Current sell position.
		sell_position = self.exchange.bid_price[0]

		# Compute the market pnl and record it.
		pnl = self.trade_qty * ( sell_position - self.market_position)
		self.market_pnl.append(timestamp, pnl)

		# Compute the market pnl as percentage.
		pnl_pct = 100 * ( sell_position / self.market_position - 1.0 )

		return pnl, pnl_pct

	def update_realized_pnl(self):
		"""Update the current realized pnl.
		"""

		# Current timestamp.
		timestamp = self.market.timestamp

		# Check the current buy and sell position.
		if self.buy_position_wealth is None or self.sell_position_wealth is None:
			pnl = 0.0
			pnl_pct = 0.0

		else:

			# Compute the unrealized pnl and record it.
			pnl = self.sell_position_wealth - self.buy_position_wealth

			# Compute the unrealized pnl as percentage and record it.
			pnl_pct = 100 * ( self.sell_position_wealth / self.buy_position_wealth - 1.0 )

			# Update buy and sell positions.
			self.buy_position_wealth = None
			self.sell_position_wealth = None

		self.realized_pnl.append(timestamp, pnl)
		self.realized_pnl_pct.append(timestamp, pnl_pct)		

		return pnl, pnl_pct
		
	def update_unrealized_pnl(self):
		"""Update the current unrealized pnl.
		"""

		# Current timestamp.
		timestamp = self.market.timestamp

		# Check the current buy position.
		if self.buy_position_price is None:
			pnl = 0.0
			pnl_pct = 0.0
		else:

			# Current sell position.
			sell_position_price = self.exchange.bid_price[0]

			# Compute the unrealized pnl and record it.
			#if self.sell_position_price is None:
			pnl = self.trade_qty * ( sell_position_price - self.buy_position_price )
			pnl -= 0.002 * pnl
			#else:
			#	pnl = -self.trade_qty * ( sell_position_price - self.sell_position_price )

		self.unrealized_pnl.append(timestamp, pnl)

		# Compute the unrealized pnl as percentage and record it
		pnl_pct = 0.0 #100 * ( sell_position / self.buy_position - 1.0 )
		self.unrealized_pnl_pct.append(timestamp, pnl_pct)

		return pnl, pnl_pct

	# ----------------------------------- #
	# --- 1. Environment user methods --- #
	# ----------------------------------- #

	def action(self, action):
		"""Act on the exchange.

		:param action: the nature of the action.
		:type action: int.
		"""

		if action is None:
			return
		else:
			self.actions.append(self.market.timestamp, float(action))

		FLAG = 0.0

		if self._action == action:
			pass
		else:
			FLAG = 1.0
			if action == 1:
				# Buy the market and 
				# update the buy position.
				self.exchange.buy_dummy_market(self.trade_qty)
				self.buy_position_price = self.exchange.ask_price[0]
				self.sell_position_price = None
				self.buy_position_wealth = self.exchange.get_wealth() 
			elif action == 0:
				# Sell the market and 
				# update the sell position.
				self.exchange.sell_dummy_market(self.trade_qty)
				self.sell_position_price = self.exchange.bid_price[0]
				self.sell_position_wealth = self.exchange.get_wealth() 
			else:
				raise ValueError("Invalid action = {}".format(action))

		self._action = action

		return FLAG

	def state(self):
		"""Returns the environment current state.
		"""
		return self.public.get_list_values()

	def done(self):
		"""Check if the simulation is done.
		"""
		done = False
		if self.public.timestamp == self.public.timestamp_last:
			done = True
		if self.market.timestamp == self.market.timestamp_last:
			done = True
		if self._wealth / self._wstart < 0.10:
			done = True
		return done

	def step(self, action=None):
		"""Step on the environment.

		:param action: name of the action.
		:type action: int.
		"""

		# Initialise the wealth.
		wealth = self._wealth

		# Perform the action.
		act = self.action(action)

		# Step on the public history.
		self.public.step(self.period)

		# Step on the market history.
		while self.public.timestamp > self.market.timestamp:
			self.market.step()

		# Update the exchange.
		self.update_exchange()

		# Get the current state and simulation status.
		state = self.state()
		done = self.done()

		# Update the unrealized and realized pnl.
		mpnl, mpnl_pct = self.update_market_pnl()
		rpnl, rpnl_pct = self.update_realized_pnl()
		upnl, upnl_pct = self.update_unrealized_pnl()		

		returns = 100 * ( self._wealth / wealth - 1.0 )
		self.returns.append(self.market.timestamp, returns)

		reward = returns 
		self.rewards.append(self.market.timestamp, reward)

		return state, done, reward
