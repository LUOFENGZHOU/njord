#!/usr/bin/env python
# coding=utf-8

import time
import random
import pandas
import datetime

from njordtoolbox import History


if __name__ == "__main__":

	# Set an empty dataframe.
	df = pandas.DataFrame()

	# Set the record.
	record = History(df, name="random")



