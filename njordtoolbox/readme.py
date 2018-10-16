#!/usr/bin/env python
# coding=utf-8


def get_title(title):
	"""Returns a title.
	"""
	return "\n### {} ###\n".format(title)

def get_text(text):
	"""Returns text.
	"""
	return "{}".format(text)

def get_text_structured(key, field, dotes=True):
	"""Returns an expression.
	"""
	if not isinstance(key, str):
		key = str(key)
	if not isinstance(field, str):
		field = str(field)
	if dotes is True:
		dotes = "." * ( 23 - len(key))
		return "{} {} = {}.".format(key, dotes, field)
	else:
		return "{} = {}.".format(key, field)

def get_dict_to_text(item):
	"""Returns a dictionnary as a structured per line text.
	"""
	save = []
	for key in item.keys():
		to_save = get_text_structured(key, item[key])
		save.append(to_save)
	return "\n".join(save)

def get_list_to_text(text):
	"""Return a str.
	"""
	save = []
	for item in text:
		if isinstance(item, str):
			to_save = get_text(item)
			save.append(to_save)
		elif isinstance(item, dict):
			to_save = get_dict_to_text(item)
			save.append(to_save)
		else:
			pass
	return "\n".join(save)

def get_tuple_to_text(text):
	"""Returns a tuple as a structured per line text.
	"""
	text = list(text)
	return get_list_to_text(text)

def get_readme(**params):
	readme = []
	for key in params.keys():

		title = get_title(key)
		field = params[key]

		if isinstance(field, str):
			expression = get_text(field)	
		elif isinstance(field, tuple):
			expression = get_tuple_to_text(field)
		elif isinstance(field, list):
			expression = get_list_to_text(field)			
		elif isinstance(field, dict):
			expression = get_dict_to_text(field)		
		else:
			expression = None

		if expression is not None:
			readme.append(title)	
			readme.append(expression)			

	return "\n".join(readme)

def save_readme(filename, *args, **kwargs):
	readme = get_readme(**kwargs)
	with open(filename, "w") as file:
		file.write(readme)	
	return

if __name__ == '__main__':
		
	# introduction
	intro = "This is a trial readme file, please be indulgent"

	# authors
	authors = {"author 1":"Ev", "author 2":"Martin", "date": "3 august"}

	# requirements
	requirements = "You should have installed pytorch"
	
	# parameters
	parameters = {}
	parameters["learning_rate"] = 0.001
	parameters["dropout"] = 0.5
	parameters["epoch"] = 100

	# model
	model = []
	model.append("*) number of neurons .... = 4")
	model.append("*) number of layers  .... = 3")
	model.append("*) activation units ..... = ReLU")

	# build the readme

	readme = {}
	readme["intro"] = intro
	readme["authors"] = authors
	readme["requirements"] = requirements
	readme["parameres"] = parameters
	readme["model"] = model

	save_readme("hello.txt", **readme)






