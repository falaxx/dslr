
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
		prediction = predict(thetas,x)
		f = open("houses.csv", 'w+')
		f.write('Index,Hogwarts House\n')
		for i in range(0, len(prediction)):
			f.write(f'{i},{prediction[i]}\n')
	except:
		print("error")
		exit(0)