#!/usr/bin/env python
# coding=utf-8

import time
import random
import pandas
import datetime

from njordtoolbox import History


if __name__ == "__main__":

	# Create an empty dataframe.
	df = []
	for i in range(10):
		data = {
			"time": datetime.datetime.now(),
			"price": random.normalvariate(100.0, 1.0)
		}
		df.append(data)
	df = pandas.DataFrame(df).set_index("time")

	# Display the dataframe.
	print(df)

	# Set the record.
	record = History(df, name="random")



