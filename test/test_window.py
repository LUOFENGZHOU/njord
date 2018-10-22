#!/usr/bin/env python
# coding=utf-8

import time
import random
import datetime

from njordtoolbox import Window


if __name__ == "__main__":

	# Set the record.
	window = Window(lookback=5)

	# Start the time.
	timer_start = time.time()

	# Fill the window with values.
	for i in range(10):
		data = {"price_avg_#t": random.normalvariate(100.0, 1.0)}
		window.append(data)
		print(window())

	# End the timer.
	timer_final = time.time()

	# Compute the elapsed time and display.
	elapsed_time = int( 1000 * ( timer_final - timer_start ) )
	print("Elapsed time = {} [ms]".format(elapsed_time))
	time.sleep(1.0)