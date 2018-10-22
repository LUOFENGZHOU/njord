#!/usr/bin/env python
# coding=utf-8

import time
import random
import pandas
import datetime

from njordtoolbox import Trades


if __name__ == "__main__":

	# Set the record.
	trades = Trades(name="trades")

	# Add trades.
	for i in range(20):

		# Set a random trade.
		timestamp = datetime.datetime.now()
		price = random.normalvariate(1.0, 0.05)
		base = random.normalvariate(100.0, 1.0)
		quote = price * base
		fees = base * 0.001
		otype = ["buy", "sell"][random.randrange(0, 2)]

		# Add the trade to the list of trades.
		trades.append(timestamp, price, base, quote, fees, otype)

	# Display the first trade.
	print("\nRetrieve the first trade:")
	print(trades.popleft())

	# Display the last trade.
	print("\nRetrieve the last trade:")	
	print(trades.pop())

	# Display the trades as pandas.
	print("\nDisplay the trades as a pandas dataframe:")
	print(trades.aspandas())

	# Display the buy trades.
	print("\nDisplay the buy trades:")
	print(trades.buy_orders())

	# Display the sell trades.
	print("\nDisplay the sell trades:")
	print(trades.sell_orders())
