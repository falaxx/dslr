import numpy as np
# all the maths fonctions needed for other files
def mean(X):
	res = 0
	for i in range (0,len(X)):
		res += X[i]
	return round((res / len(X)),6)

def min(X):
	res = 1000000.0
	for i in range (0,len(X)):
		if X[i] < res:
			res = X[i]
	return round(res,6)

def max(X):
	res = -1000000.0
	for i in range (0,len(X)):
		if X[i] > res:
			res = X[i]
	return round(res,6)

def std(X):
	m = mean(X)
	res = 0
	for i in range (0,len(X)):
		res += (X[i] - m) ** 2
	res = (res / len(X)) ** 0.5
	return round(res,6)

def percentile(X, percent):
	X.sort()
	rank = percent * (len(X) - 1)
	f = np.floor(rank)
	c = np.ceil(rank)
	if f == c:
		return round(X[int(rank)],6)
	return round((X[int((c + f) / 2)]), 6)