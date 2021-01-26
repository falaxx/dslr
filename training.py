
#We are importing all necessary libraries to implement our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle


def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')

class LogObj(object):

	def __init__(Logreg, lr=0.1, iteration=1000):
		Logreg.lr=lr
		Logreg.iteration=iteration

	def sigmoid_(Logreg,x):
		return 1 / (1 + np.exp(-x))

	def cost_(Logreg,h,theta,y):
		return (1 / len(y)) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))))
	
	def gradient_descent_(Logreg,x,h,theta,y,m,):
		return  (theta - Logreg.lr * np.dot(x.T, (h - y)) / m)
	
	def fit(Logreg,x,y):
		print("Processing data..")
		Logreg.theta = []
		Logreg.cost = []
		x = np.insert(x, 0, 1, axis=1)
		m = len(y)
		for i in np.unique(y):
			y_1vsall = np.where(y == i, 1, 0)
			theta = np.zeros(x.shape[1])
			for _ in range(Logreg.iteration):
				z = x.dot(theta)
				h = Logreg.sigmoid_(z)
				# print (z)
				theta = Logreg.gradient_descent_(x,h,theta,y_1vsall,m)
			Logreg.theta.append((theta, i))
		print("model trained !")
		return Logreg


	def predict(Logreg,x):
		x = np.insert(x, 0, 1, axis=1)
		prediction = list()
		for i in x:
				prediction.append(max((Logreg.sigmoid_(i.dot(theta)), c) for theta, c in Logreg.theta)[1])
		return prediction

	def score(Logreg,x, y):
		print (sum(Logreg.predict(x) == y))
		print(len(y))
		return (sum(Logreg.predict(x) == y) / len(y))

if __name__ == '__main__':
	# try:
		# parsing
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="dataset")
	args = parser.parse_args()
	filename = args.dataset
	data = pd.read_csv(filename,dtype=str)
	data = data.fillna(0)
	data.dtypes
	data.columns = ['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
	olddata_y = np.copy(data['Hogwarts House'].values)
	olddata_x = np.copy(data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand',],axis=1).values)
	y_data = data['Hogwarts House'].values
	x = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand',],axis=1).values
	# scaling
	scaler = StandardScaler()
	x = scaler.fit_transform(x)
	olddata_x = scaler.fit_transform(olddata_x)
	print(x)
	print(y_data)
	scores = []
	#regression
	logi = LogObj(iteration=1000)
	logi = logi.fit(x, y_data)
	print("Predictions in progress..")
	score1 = logi.score(olddata_x,olddata_y)
	print("the accuracy of the model is ",score1)
	predictionall = logi.predict(olddata_x)
	# print(predictionall)
	print(len(predictionall),len(olddata_y))
	print("accuracy sklearn = " + str(accuracy_score(olddata_y, predictionall)))
	f = open("weights.csv", "w")
	for i in range(0,4):
		for j in range(0,len(logi.theta[i][0])):
			f.write(decimal_str(logi.theta[i][0][j]))	
			f.write(" ")
		f.write("\n")
	f.close()

	# except:
	# 	print("error")
	# 	exit(0)