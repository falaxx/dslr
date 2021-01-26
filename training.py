
#We are importing all necessary libraries to implement our model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score
from utils import LogObj,std_,mean_

def decimal_str(x: float, decimals: int = 10) -> str:
	return format(x, f".{decimals}f").lstrip().rstrip('0')


if __name__ == '__main__':
	try:
		# parsing
		parser = argparse.ArgumentParser()
		parser.add_argument("dataset", type=str, help="dataset")
		args = parser.parse_args()
		filename = args.dataset
		data = pd.read_csv(filename,dtype=str)
		data = data.fillna(0)
		data.dtypes
		cols = ['Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
		data.columns = ['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand','Arithmancy','Astronomy','Herbology','Defense Against the Dark Arts','Divination','Muggle Studies','Ancient Runes','History of Magic','Transfiguration','Potions','Care of Magical Creatures','Charms','Flying']
		print("Scaling data..")
		for col in cols:
			data[col] = data[col].apply(lambda x: float(x) if x == x else "")
			std = std_(data[col])
			mean = mean_(data[col])
			data[col] = (data[col] - mean) / std
		y = data['Hogwarts House'].values
		x = data.drop(['Index','Hogwarts House','First Name','Last Name','Birthday','Best Hand',],axis=1).values
		logi = LogObj(iteration=1000)
		logi = logi.fit(x, y)
		print("Predictions in progress..")
		predictionall = logi.predict(x)
		print("The accuracy I calculated for the model is "+ str(sum(logi.predict(x) == y) / len(y)))
		print("Accuracy from sklearn = " + str(accuracy_score(y, predictionall)))
		f = open("weights.csv", "w")
		for i in range(0,4):
			for j in range(0,len(logi.theta[i][0])):
				f.write(decimal_str(logi.theta[i][0][j]))	
				f.write(" ")
			f.write("\n")
		f.close()
		print("Weights saved in csv")
		i = 0
		legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		for cost,c in logi.cost:
			if i == 0:
				plt.plot(range(len(cost)),cost,'r')
			if i == 1:
				plt.plot(range(len(cost)),cost,'y')
			if i == 2:
				plt.plot(range(len(cost)),cost,'b')
			if i == 3:
				plt.plot(range(len(cost)),cost,'g')
			i+=1
			plt.title("Cost over iterations")
			plt.xlabel("Iterations")
			plt.ylabel("Cost")
		plt.legend(legend, loc='upper right', frameon=False)
		plt.show()
	except:
		print("error")
		exit(0)