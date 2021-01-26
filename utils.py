import numpy as np
import csv
# all the fonctions needed for other files

def mean_(x):
	res = 0
	for i in range (0,len(x)):
		res += x[i]
	return round((res / len(x)),6)

def min_(x):
	res = 1000000.0
	for i in range (0,len(x)):
		if x[i] < res:
			res = x[i]
	return round(res,6)

def max_(x):
	x = list(x)
	res = -1000000.0
	for i in range (0,len(x)):
		if x[i] > res:
			res = x[i]
	return round(res,6)

def std_(x):
	m = mean_(x)
	res = 0
	for i in range (0,len(x)):
		res += (x[i] - m) ** 2
	res = (res / len(x)) ** 0.5
	return round(res,6)

def percentile_(x, percent):
	x.sort()
	rank = percent * (len(x) - 1)
	f = np.floor(rank)
	c = np.ceil(rank)
	if f == c:
		return round(x[int(rank)],6)
	return round((x[int((c + f) / 2)]), 6)

class LogObj(object):

	def __init__(Logreg, lr=0.1, iteration=1000):
		Logreg.lr=lr
		Logreg.iteration=iteration

	def sigmoid_(Logreg,x):
		return 1 / (1 + np.exp(-x))
	
	def gradient_descent_(Logreg,x,h,theta,y,m,):
		return  (theta - Logreg.lr * np.dot(x.T, (h - y)) / m)
	
	def cost_(Logreg,h,theta, y):
		return (1 / len(y)) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
	
	def fit_(Logreg,x,y):
		print("Processing data..")
		Logreg.theta = []
		Logreg.cost = []
		x = np.insert(x, 0, 1, axis=1)
		m = len(y)
		for i in np.unique(y):
			cost = []
			y_1vsall = np.where(y == i, 1, 0)
			theta = np.zeros(x.shape[1])
			for _ in range(Logreg.iteration):
				z = x.dot(theta)
				h = Logreg.sigmoid_(z)
				theta = Logreg.gradient_descent_(x,h,theta,y_1vsall,m)
				cost.append(Logreg.cost_(h,theta,y_1vsall)) 
			Logreg.theta.append((theta, i))
			Logreg.cost.append((cost,i))
		print("model trained !")
		return Logreg

	def predict(Logreg,x):
		x = np.insert(x, 0, 1, axis=1)
		prediction = list()
		for i in x:
			res2 = ""
			res = 0
			for theta, c in Logreg.theta:
				if Logreg.sigmoid_(i.dot(theta)) > res:
					res = Logreg.sigmoid_(i.dot(theta))
					res2 = c
			prediction.append(res2)
		return prediction


def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')

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
