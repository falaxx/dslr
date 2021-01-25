import numpy as np
import csv

# all the fonctions needed for other files

def load_csv(filename):
	dataset = list()
	try:
		with open(filename) as csvfile:
			reader = csv.reader(csvfile)
			for _ in reader:
				row = list()
				for value in _:
					try:
						value = float(value)
					except:
						if not value:
							value = np.nan
					row.append(value)
				dataset.append(row)
		return np.array(dataset, dtype=object)
	except:
		print("error")
		exit(0)

def mean_(X):
	res = 0
	for i in range (0,len(X)):
		res += X[i]
	return round((res / len(X)),6)

def min_(X):
	res = 1000000.0
	for i in range (0,len(X)):
		if X[i] < res:
			res = X[i]
	return round(res,6)

def max_(X):
	res = -1000000.0
	for i in range (0,len(X)):
		if X[i] > res:
			res = X[i]
	return round(res,6)

def std_(X):
	m = mean_(X)
	res = 0
	for i in range (0,len(X)):
		res += (X[i] - m) ** 2
	res = (res / len(X)) ** 0.5
	return round(res,6)

def percentile_(X, percent):
	X.sort()
	rank = percent * (len(X) - 1)
	f = np.floor(rank)
	c = np.ceil(rank)
	if f == c:
		return round(X[int(rank)],6)
	return round((X[int((c + f) / 2)]), 6)