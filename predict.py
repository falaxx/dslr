
#We are importing all necessary libraries to implement our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')


def sigmoid_(x):
		return 1 / (1 + np.exp(-x))

def predict(thetas,x):
		x = np.insert(x, 0, 1, axis=1)
		prediction = list()
		for i in x:
				prediction.append(max((sigmoid_(i.dot(theta)), c) for theta, c in thetas)[1])
		print(prediction)
		return prediction


if __name__ == '__main__':
	try:
# parsing
		parser = argparse.ArgumentParser()
		parser.add_argument("dataset", type=str, help="dataset")
		parser.add_argument("weight", type=str, help="weight file")

		args = parser.parse_args()
		filedata = args.dataset
		fileweight = args.weight
		data = pd.read_csv(filedata,dtype=str)
		weight = np.loadtxt(fileweight, dtype = np.float64)
		data.dtypes
		data.columns = ['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
		y_data = data['Hogwarts House'].values  #segregating the label value from the feature value.
		x = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand',],axis=1).values
		# scaling
		scaler = StandardScaler()
		x= scaler.fit_transform(x)
		# print(x)
		thetas= list()
		thetas.append((weight[0], "Gryffindor"))
		thetas.append((weight[1], "Hufflepuff"))
		thetas.append((weight[2], "Ravenclaw"))
		thetas.append((weight[3], "Slytherin"))
		print(thetas)
		prediction = predict(thetas,x)
		f = open("houses.csv", 'w+')
		f.write('Index,Hogwarts House\n')
		for i in range(0, len(prediction)):
			f.write(f'{i},{prediction[i]}\n')
		print(len(prediction))
		print(len(data))
		# scores = []

		# # for _ in range (1):
		# x_train,x_test,y_train,y_test = train_test_split(x,y_data,test_size = 0.30)
		# logi = LogObj(iteration=1000)
		# logi = logi.fit(x_train, y_train)
		# print("Predictions in progress..")
		# predition1 = logi.predict(x_test)
		# print("predictions done!")
		# print("Calculating accuracy..")
		# score1 = logi.score(x_test,y_test)
		# print("the accuracy of the model is ",score1)
		# scores.append(score1)
		# print(np.mean(scores))
		# print(logi.theta)

		# f = open("weights.csv", "w")
		# for i in range(0,4):
		# 	for j in range(0,len(logi.theta[i][0])):
		# 		# if  np.char.isnumeric(logi.theta[i][0][j])==True:
		# 		print((logi.theta[i][0][j]))
		# 		f.write(decimal_str(logi.theta[i][0][j]))	
		# 		f.write(" ")
		# 	f.write("\n")

		# f.close()

	except:
		print("error")
		exit(0)