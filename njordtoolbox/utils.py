#!/usr/bin/env python
# coding=utf-8

import njordtoolbox


def dataset_filename(dataset, symbol, period, version):
	"""Returns the filename convention for a njord dataset.

	:param dataset: the name of the dataset.
	:type dataset: str.
	:param symbol: the name of the symbol.
	:type symbol: str.
	:param period: the period length.
	:type period: int.
	:param version: the dataset mode (either train, dev or test).
	:type vesion: str.
	"""
	filename = []
	filename.append("dataset_{}".format(dataset))
	filename.append("symbol_{}".format(symbol))
	filename.append("period_{}".format(period))
	filename.append("{}".format(version))
	return "_".join(filename)

def load_dataset(platform, symbol, dataset, period, version, location):
	"""Load the dataset from the specified location.

	:param dataset: the name of the dataset.
	:type dataset: str.
	:param symbol: the name of the symbol.
	:type symbol: str.
	:param period: the period length.
	:type period: int.
	:param version: the dataset mode (either train, dev or test).
	:type vesion: str.
	:param location: location where the file can be found (optional).
	:type location: str.
	"""
	handler = njordtoolbox.FileHandler("home")
	handler.path_append(location, "dataset", platform, symbol)
	filename = dataset_filename(dataset, symbol, period, version)	
	return handler.load_pandas_timeseries(filename)	

def load_trained_model(save_path, location):
	"""Load the model from the specified location.

	:param save_path: the list of directories that form the path.
	:type save_path: list.
	:param location: the save location of the model (optional).
	:type location: str.	
	"""
	handler = njordtoolbox.FileHandler("home")
	handler.path_append(location, "supervised", "model", save_path)
	batch = handler.load_pickle("train_batch")	
	model = handler.load_pickle("model")
	weights = handler.load_torch("weights")
	return (batch, model, weights)

