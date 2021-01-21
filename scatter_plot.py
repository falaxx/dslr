import numpy as np
import matplotlib.pyplot as plt
from describe import describe,load_csv
from utils import percentile, mean,std

def scatter_plot(X, y, legend, xlabel, ylabel, show):

	# rempli et save les percentile 	 
	h1 = X[:327][~np.isnan(X[:327])]
	h2 = X[327:856][~np.isnan(X[327:856])]
	h3 = X[856:1299][~np.isnan(X[856:1299])]
	h4 = X[1299:][~np.isnan(X[1299:])]
	h1y = y[:327][~np.isnan(y[:327])]
	h2y = y[327:856][~np.isnan(y[327:856])]
	h3y = y[856:1299][~np.isnan(y[856:1299])]
	h4y = y[1299:][~np.isnan(y[1299:])]
	
	dataset = list()
	dataset.append(percentile(h1,0.50)/percentile(h1y,0.50))
	dataset.append(percentile(h2,0.50)/percentile(h2y,0.50))
	dataset.append(percentile(h3,0.50)/percentile(h3y,0.50))
	dataset.append(percentile(h4,0.50)/percentile(h4y,0.50))
	print("deviation of median" + str(xlabel)+ str(ylabel))
	print(std(dataset))

	if show == 1:
		plt.scatter(X[:327], y[:327], color='red', alpha=0.5)
		plt.scatter(X[327:856], y[327:856], color='yellow', alpha=0.5)
		plt.scatter(X[856:1299], y[856:1299], color='blue', alpha=0.5)
		plt.scatter(X[1299:], y[1299:], color='green', alpha=0.5)
		plt.legend(legend, loc='upper right', frameon=False)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()
	
	return std(dataset)

if __name__ == '__main__':
	dataset = load_csv('./datasets/dataset_train.csv')
	data = dataset[1:, :]
	data = data[data[:, 1].argsort()]
	save = 1000000.0
	res = 6,7

	for i in range(6, 19):
		for j in range(7, 19):
			# un graph par combinaison de matiere
			X = np.array(data[:, i], dtype=float)
			y = np.array(data[:, j], dtype=float)
			legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
			if (dataset[0, i] != dataset[0, j]):
				k = scatter_plot(X, y, legend=legend, xlabel=dataset[0, i], ylabel=dataset[0, j], show=0)
				if (k < save):
					save = k
					res = i,j
	X = np.array(data[:, res[0]], dtype=float)
	y = np.array(data[:, res[1]], dtype=float)
	legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
	scatter_plot(X, y, legend=legend, xlabel=dataset[0, res[0]], ylabel=dataset[0, res[1]], show=1)
			

  