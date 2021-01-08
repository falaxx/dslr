import numpy as np
import matplotlib.pyplot as plt
from describe import describe,load_csv
from utils import percentile, mean

def scatter_plot(X, y, legend, xlabel, ylabel, show):

	# rempli et save les percentile 	 
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
	if show == 1:
		plt.scatter(X[:327], y[:327], color='red', alpha=0.5)
		plt.scatter(X[327:856], y[327:856], color='yellow', alpha=0.5)
		plt.scatter(X[856:1299], y[856:1299], color='blue', alpha=0.5)
		plt.scatter(X[1299:], y[1299:], color='green', alpha=0.5)
		plt.legend(legend, loc='upper right', frameon=False)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

if __name__ == '__main__':
	dataset = load_csv('./datasets/dataset_train.csv')
	data = dataset[1:, :]
	data = data[data[:, 1].argsort()]
	

	for i in range(6, 19):
		for j in range(7, 19):
			# un graph par combinaison de matiere
			X = np.array(data[:, i], dtype=float)
			y = np.array(data[:, j], dtype=float)
			legend = ['Grynffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
			if (dataset[0, i] != dataset[0, j]):
				scatter_plot(X, y, legend=legend, xlabel=dataset[0, i], ylabel=dataset[0, j], show=1)

  