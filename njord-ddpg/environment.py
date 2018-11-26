#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import ast
import math
import numpy
import pandas

# Import third-party packages.
from njordtoolbox import Record
from njordtoolbox import History
from njordtoolbox import Trades


class Environment:
	"""Class that handles a trading environment for RL.

	:attr action: the current action taken between -1 and +1.
	:type action: float.
	:attr wealth: the current wealth.
	:type wealth: float.
	:attr quote_qty: the initial quote qty or wealth.
	:type quote_qty: float.
	:attr trade_qty: the qty to trade.
	:type trade_qty: float.
	:attr public: the dataset.
	:type public: History object.
	"""

	def __init__(self, filename, quote_qty, trade_qty, fees=0.001):
		"""Special method for class initialisation.

		:param filename: the name of file to download.
		:type filename: str.
		:param quote_qty: the initial quote_qty.
		:type quote_qty: float.
		:param trade_qty: the qty to trade.
		:type trade_qty: float.
		"""

		# Set the default attributes.
		self.fees = fees
		self.quote_qty = quote_qty
		self.trade_qty = trade_qty	

		# Set the variable attributes.
		self._action = 0.0
		self._wealth = quote_qty

		# Set the public history.
		df = pandas.read_csv(filename)
		df = df.set_index("time")
		df.index = pandas.to_datetime(df.index)
		self.public = History(df)
		self.public.scope("close_price")

		# Set the records.
		self.actions = Record("actions")
		self.balance = Record("balance")
		self.rewards = Record("rewards")

		return

	def __repr__(self):
		"""Special method for class object representation.
		"""
		_repr = {
			"fees": self.fees,
			"action": self.action,
			"wealth": self.wealth,
			"quote_qty": self.quote_qty,
			"trade_qty": self.trade_qty
		}
		return _repr

	def __str__(self):
		"""Special method for class object printable version.
		"""
		_str = []
		for a, b in self.__repr__().items():
			_str.append("{} = {}".format(a, b))
		return "{}({})".format(self.__class__.__name__, ", ".join(_str))
	
	# ------------------------------------ #
	# --- 0. Environment basic methods --- #
	# ------------------------------------ #

	def clear(self):
		"""Clear the environment from the memory.
		"""
		self.actions.clear()
		self.balance.clear()
		self.rewards.clear()
		self.public.clear()		
		return

	def reset(self, random=True):
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

		# Set the default attributes.
		self._action = 0.0
		self._wealth = self.quote_qty

		# Clear the environment.
		self.clear()

		return		

	# ----------------------------------- #
	# --- 1. Environment user methods --- #
	# ----------------------------------- #

	def action(self, action):
		"""Performs an action on the exchange.

		:param action: the action value between -1.0 and +1.0.
		:type action: float.
		"""

		# Check if action is valid and record it.
		if action is None:
			return
		else:
			self.actions.append(self.public.timestamp, float(action))	

		# Buy or sell depending on the action value.
		if self._action >= 0.0 and action >= 0.0:
			pass
		elif self._action <= 0.0 and action <= 0.0:
			pass
		elif self._action <= 0.0 and action > 0.0:
			pass
		elif self._action >= 0.0 and action < 0.0:
			pass
		else:
			msg = "Invalid action = {}".format(action)
			raise ValueError(msg)

		# Update the current action.
		self._action = action

		return

	def state(self):
		"""Returns the current state of the environment.

		:return: the current state for the public historical data.
		:rtype: dict.
		"""
		return self.public.asdict()

	def done(self):
		"""Check if the simulation is done.

		:return: True if simulation is done, False otherwise.
		:rtype: Boolean.
		"""
		if self.public.timestamp == self.public.timestamp_last:
			return True
		elif self.public.timestamp == self.public.timestamp_last:
			return True
		elif self._wealth / self.quote_qty < 0.01:
			return True
		else:
			return False

	def reward(self, price, action, prev_price, prev_action):
		"""Returns the reward function.

		:param price: the current price.
		:type price: float.
		:param action: the current action.
		:type action: float.
		:param prev_price: the previous price.
		:type prev_price: float.
		:param prev_action: the previous action.
		:type prev_action: float.

		:return: the additive reward.
		:type: float.
		"""
		r_p = action * ( price - prev_price ) / prev_price
		r_n = self.fees * math.sqrt( ( action - prev_action ) ** 2 )
		return 100 * ( r_p - r_n )

	def step(self, action=None):
		"""Step on the environment.

		:param action: the action to be performed.
		:type action: float.

		:return state: the next public state.
		:rtype state: list<float>.
		:return done: True if the simulation is done, False otherwise.
		:rtype done: bool.
		:return reward: the reward associated to the action.
		:rtype reward: float.
		"""

		# Register previous price and action.
		prev_price = self.public.asdict()["close_price_#t"]
		prev_action = self._action

		# Perform the action.
		act = self.action(action)

		# Step on the public history.
		self.public.step()

		# Get the current state and simulation status.
		state = self.state()
		done = self.done()

		# Register the new price.
		price = state["close_price_#t"]

		# Compute the new price.
		reward = self.reward(price, action, prev_price, prev_action)

		# Record the rewards.	
		self.rewards.append(self.public.timestamp, reward)

		# Return the state
		return state, done, reward
