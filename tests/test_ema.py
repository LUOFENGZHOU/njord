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
	atlas.add_indicator("EMA",  4, label="ema_4_close_price_#t")
	atlas.add_indicator("EMA", 8, label="ema_8_close_price_#t")
	
	# Reset the environment.
	env.reset(quote_qty=QUOTE_QTY, random=START_RANDOM)

	# Set the counter.
	counter = 0

	# Set the registers.
	ema_4 = njordtoolbox.Record("EMA_4")
	ema_8 = njordtoolbox.Record("EMA_8")
	print(ema_4)
	print(type(ema_4))

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
		ema_4.append(env.public.timestamp, info["ema_4_close_price_#t"])
		ema_8.append(env.public.timestamp, info["ema_8_close_price_#t"])

	# ------------ #
	# --- PLOT --- #
	# ------------ #

	fig, ax = plt.subplots(2, sharex=True)

	style = {"linewidth":0.75, "marker":".", "markersize":2.0}

	# plot 1.
	open_price = env.public.records["close_price"]
	close_price = env.public.records["close_price"]
	ax[0].plot(close_price.t, close_price.x, color="r", **style)
	ax[0].plot(ema_4.t, ema_4.x, color="b", **style)
	ax[0].plot(ema_8.t, ema_8.x, color="g", **style)
	ax[0].set_xlabel("price")
	ax[0].grid()

	# plot 2.
	ax[1].plot(ema_4.t, ema_4.x, color="b", **style)
	ax[1].plot(ema_8.t, ema_8.x, color="g", **style)
	ax[1].set_ylabel("moving average")
	ax[1].grid()

	plt.gcf().autofmt_xdate()
	plt.show()


