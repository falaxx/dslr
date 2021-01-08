import argparse
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from describe import describe,load_csv
from utils import percentile, mean, std

def histogram(X, legend, title, xlabel, ylabel, show):
	#  divise la data en 4 ecoles
	h1 = X[:327]
	h1 = h1[~np.isnan(h1)]
	h1mean = percentile(h1,0.50)

	h2 = X[327:856]
	h2 = h2[~np.isnan(h2)]
	h2mean = percentile(h2,0.50)

	h3 = X[856:1299]
	h3 = h3[~np.isnan(h3)]
	h3mean = percentile(h3,0.50)

	h4 = X[1299:]
	h4 = h4[~np.isnan(h4)]
	h4mean = percentile(h4,0.50)

	# cherche le meilleur combo


	dataset = list()
	dataset.append(h1mean)
	dataset.append(h2mean)
	dataset.append(h3mean)
	dataset.append(h4mean)
	print(title)
	print("deviation of median")
	print(std(dataset))

	# montre le graph choisi
	if show == 1:
		plt.hist(h1, color='red', alpha=0.5, stacked = True)
		plt.hist(h2, color='yellow', alpha=0.5, stacked = True)
		plt.hist(h3, color='blue', alpha=0.5, stacked = True)
		plt.hist(h4, color='green', alpha=0.5, stacked = True)
		plt.legend(legend, loc='upper right', frameon=False)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	return std(dataset)

	

if __name__ == '__main__':
	dataset = load_csv('./datasets/dataset_train.csv')
	try:
		data = dataset[1:, :]
		data = data[data[:, 1].argsort()]
		# j = 1000000.0
		save = 1000000.0
		res = 6
		# test toutes les categories
		for i in range(6, 19):
			X = np.array(data[:, i], dtype=float)
			legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
			j = histogram(X, legend=legend, title=dataset[0, i], xlabel='Marks', ylabel='Number of student',show=0)
			if (j < save):
				save = j
				res = i
		X = np.array(data[:, res], dtype=float)
		legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		# montre que la bonne
		histogram(X, legend=legend, title=dataset[0, res], xlabel='Marks', ylabel='Number of student', show=1)


	except:
		exit()

	