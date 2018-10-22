#!/usr/bin/env python
# coding=utf-8

import time
import random
import datetime

from njordtoolbox import Record


if __name__ == "__main__":

	# Set the record.
	record = Record(name="random")

	# Feed the record.
	for i in range(10):

		# Sleep ... 
		time.sleep(0.1)

		# Record a timestamp t and a value x
		t = datetime.datetime.now()
		x = random.normalvariate(0.0, 1.0)
		record.append(t, x)

	# Display the record as a numpy array,
	print("Get the record as numpy.")
	print(record.asnumpy())

	# Display the record as pandas dataframe.
	print("Get the record as pandas.")
	print(record.aspandas())
