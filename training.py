
#We are importing all necessary libraries to implement our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')

class LogObj(object):

	def __init__(Logreg, lr=0.01, iteration=1000):
		Logreg.lr=lr
		Logreg.iteration=iteration

	def sigmoid_(Logreg,x):
		return 1 / (1 + np.exp(-x))

	def cost_(Logreg,h,theta,y):
		return ((1 / len(y)) * (np.sum(-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h)))))
	
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
				theta = Logreg.gradient_descent_(x,h,theta,y_1vsall,m)
			Logreg.theta.append((theta, i))
		print("model trained !")
		return Logreg

	def predict(Logreg,x):
		x = np.insert(x, 0, 1, axis=1)
		prediction = [max((Logreg.sigmoid_(i.dot(theta)), c) for theta, c in Logreg.theta)[1] for i in x ]
		return prediction

	def score(Logreg,X, y): #This function compares the predictd label with the actual label to find the model performance
		return (sum(Logreg.predict(X) == y) / len(y))

if __name__ == '__main__':
# try:
# parsing
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="dataset")
	args = parser.parse_args()
	filename = args.dataset
	data = pd.read_csv(filename,dtype=str)
	data.dtypes
	data.columns = ['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
	data = data.dropna()
	y_data = data['Hogwarts House'].values  #segregating the label value from the feature value.
	x = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand',],axis=1).values
	# scaling
	scaler = StandardScaler()
	x= scaler.fit_transform(x)
	print(x)
	scores = []


	# for _ in range (1):
	x_train,x_test,y_train,y_test = train_test_split(x,y_data,test_size = 0.30)
	logi = LogObj(iteration=1000)
	logi = logi.fit(x_train, y_train)
	print("Predictions in progress..")
	predition1 = logi.predict(x_test)
	print("predictions done!")
	print("Calculating accuracy..")
	score1 = logi.score(x_test,y_test)
	print("the accuracy of the model is ",score1)
	scores.append(score1)
	print(np.mean(scores))
	print(logi.theta)

	# f = open("weights.csv", "w")


	save_object(logi,"weights.csv")
	# save('data.csv', np.array(logi.theta))
	# for k in range(0,len(logi.theta)):
	# 	for i in range(0,len(logi.theta[k])):
	# 		for j in range (0,len(logi.theta[k][i])):
	# 			if i%2 ==0:
	# 				f.write(decimal_str(logi.theta[k][i][j]))
	# 				f.write(",")
	# 		f.write("\n")
		
	# f.close()
	
		# y_1vsall = np.where(y == i, 1, 0)
		# theta = np.zeros(x.shape[1])
		# for _ in range(Logreg.iteration):
		# 	z = x.dot(theta)
		# 	h = Logreg.sigmoid_(z)
		# 	theta = Logreg.gradient_descent_(x,h,theta,y_1vsall,m)
		# Logreg.theta.append((theta, i))
	# except:
	#     print("error")
	#     exit(0)