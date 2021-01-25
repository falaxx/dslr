import argparse
import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils import percentile_,std_,load_csv

def histogram(X, legend, title, xlabel, ylabel, show):
	#  divise la data en 4 ecoles
	h1 = X[:327][~np.isnan(X[:327])]
	h2 = X[327:856][~np.isnan(X[327:856])]
	h3 = X[856:1299][~np.isnan(X[856:1299])]
	h4 = X[1299:][~np.isnan(X[1299:])]
	
	dataset = list()
	dataset.append(percentile_(h1,0.50))
	dataset.append(percentile_(h2,0.50))
	dataset.append(percentile_(h3,0.50))
	dataset.append(percentile_(h4,0.50))
	print(title)
	print("deviation of median")
	print(std_(dataset))

	# montre le graph choisi
	if show == 1:
		plt.hist(h1, color='red', alpha=0.3, stacked = True)
		plt.hist(h2, color='yellow', alpha=0.3, stacked = True)
		plt.hist(h3, color='blue', alpha=0.3, stacked = True)
		plt.hist(h4, color='green', alpha=0.3, stacked = True)
		plt.legend(legend, loc='upper right', frameon=False)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
	return std_(dataset)

	

if __name__ == '__main__':
	try:
		dataset = load_csv('./datasets/dataset_train.csv')
		data = dataset[1:, :]
		data = data[data[:, 1].argsort()]
		# j = 1000000.0
		save = 1000000.0
		res = 6
		# test toutes les categories
		for i in range(6, 19):
			X = np.array(data[:, i], dtype=float)
			legend = ['Ravenclaw', 'Slytherin','Gryffindor', 'Hufflepuff']
			j = histogram(X, legend=legend, title=dataset[0, i], xlabel='Grades', ylabel='Students',show=0)
			if (j < save):
				save = j
				res = i
		X = np.array(data[:, res], dtype=float)
		legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		# montre que la bonne
		histogram(X, legend=legend, title=dataset[0, res], xlabel='Grades', ylabel='Students', show=1)

	except:
		print("error")
		exit()

	