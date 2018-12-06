#!/usr/bin/env python
# coding=utf-8

import time
import random
import datetime

from njord import Window


if __name__ == "__main__":

	# Set the record.
	window = Window(lookback=5)

	# Set normalisation
	window.add_norm("#t", "pct_change", ref="price_avg_#t")
	print(window)

	# Start the time.
	timer_start = time.time()

	# Fill the window with values.
	for i in range(10):

		# Create a random data sample.
		data = {
			"price_avg_#t": random.normalvariate(100.0, 1.0),
			"price_low_#t": random.normalvariate(90.0, 1.0),
			"price_high_#t": random.normalvariate(110, 1.0)
		}

		# Append the data sample to the window.
		window.append(data)

		# Display the most recent samples.
		print(window())

	# End the timer.
	timer_final = time.time()

	# Compute the elapsed time and display.
	elapsed_time = int( 1000 * ( timer_final - timer_start ) )
	print("Elapsed time = {} [ms]".format(elapsed_time))
	time.sleep(1.0)