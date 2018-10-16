#!/usr/bin/env python
# coding=utf-8

# Import built-in packages.
import ast
import math
import numpy
import pandas
import matplotlib.pyplot as plt


# Import third party packages.
import njordtoolbox

# Set the environment hyper parameters.
LOCATION = "Bor/njord"
PLATFORM = "binance"
SYMBOL = "BTCUSDT"
PUBLIC = "Binance_1"
MARKET = "Market"
VERSION = "dev"
PERIOD = 60
ENV_PERIOD = 600
QUOTE_QTY = 10000.0
START_RANDOM = False

# Set the atlas hyper parameters.
PAST = 30

if __name__ == "__main__":
	
	# Load the public dataset.
	public = njordtoolbox.utils.load_dataset(PLATFORM, SYMBOL, PUBLIC, PERIOD, VERSION, LOCATION)

	# Load the market dataset.
	market = njordtoolbox.utils.load_dataset(PLATFORM, SYMBOL, MARKET, PERIOD, VERSION, LOCATION)

	# Load the environment.
	env = njordtoolbox.Environment(PLATFORM, SYMBOL, public, market, ENV_PERIOD)

	# Load the data generator.
	atlas = njordtoolbox.Atlas(col=env.public.df.columns, lk=ENV_PERIOD)
	atlas.add_indicator("EMA",  9, marker="#t")
	atlas.add_indicator("EMA", 15, marker="#t")
	atlas.add_indicator("RSI", 21, marker="#n")
	print(atlas)
	
	# Reset the environment.
	env.reset(quote_qty=QUOTE_QTY, random=START_RANDOM)

	# Set the counter.
	counter = 0

	# Set the registers.
	ema_9 = njordtoolbox.Record("EMA_9")
	ema_15 = njordtoolbox.Record("EMA_15")
	rsi = njordtoolbox.Record("RSI")

	# Loop on the environment.
	for i in range(0,300):

		counter += 1

		# Select and perform an action on the environment.
		next_state, done, _ = env.step()

		# Observe new state.
		if next_state is not None:
			atlas.append(next_state)
		else:
			atlas.clear()

		# Display the last RSI.
		info = atlas.last()

		# Record the inforamtion.
		ema_9.append(env.public.timestamp, info["EMA_9_#t"])
		ema_15.append(env.public.timestamp, info["EMA_15_#t"])
		rsi.append(env.public.timestamp, info["RSI_21_#n"])

	# ------------ #
	# --- PLOT --- #
	# ------------ #

	fig, ax = plt.subplots(2, sharex=True)

	style = {"linewidth":0.75, "marker":".", "markersize":2.0}

	# plot 1.
	open_price = env.public.records["close_price"]
	close_price = env.public.records["close_price"]
	ax[0].plot(close_price.t, close_price.x, color="r", **style)
	ax[0].plot(ema_9.t, ema_9.x, color="b", **style)
	ax[0].plot(ema_15.t, ema_15.x, color="g", **style)
	ax[0].set_xlabel("price")
	ax[0].grid()

	# plot 2.
	ax[1].plot(rsi.t, rsi.x, color="b", **style)
	ax[1].set_ylabel(rsi.name)
	ax[1].grid()

	plt.gcf().autofmt_xdate()
	plt.show()


