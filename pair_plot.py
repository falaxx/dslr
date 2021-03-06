import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils import load_csv,exit_

def scatter(ax, x, y):
  ax.scatter(x[:327], y[:327], s=1, color='red', alpha=0.5)
  ax.scatter(x[327:856], y[327:856], s=1, color='yellow', alpha=0.5)
  ax.scatter(x[856:1299], y[856:1299], s=1, color='blue', alpha=0.5)
  ax.scatter(x[1299:], y[1299:], s=1, color='green', alpha=0.5)

def hist(ax, X):

	h1 = X[:327][~np.isnan(X[:327])]
	h2 = X[327:856][~np.isnan(X[327:856])]
	h3 = X[856:1299][~np.isnan(X[856:1299])]
	h4 = X[1299:][~np.isnan(X[1299:])]
	ax.hist(h1, alpha=0.5)
	ax.hist(h2, alpha=0.5)
	ax.hist(h3, alpha=0.5)
	ax.hist(h4, alpha=0.5)

def pair_plot(dataset,features,legend):
	try:
		_, ax = plt.subplots(nrows=13, ncols=13,gridspec_kw={'hspace': 0, 'wspace': 0})
		for row in range(6, 19):
			for col in range(6, 19):
				# one subplot/combinaison
				ax[row-6, col-6].tick_params(labelbottom=False,labelleft=False)
				x = np.array(dataset[:, col], dtype=float)
				y = np.array(dataset[:, row], dtype=float)
				if col == row:
					ax[row-6, col-6].tick_params(reset='True')
					ax[row-6, col-6].tick_params(labelbottom=False,labelleft=False)
					hist(ax[row-6, col-6], y)
				else:
					scatter(ax[row-6,col-6], x, y)
				if ax[row-6, col-6].is_last_row():
					ax[row-6, col-6].set_xlabel(features[col-6].replace(' ', '\n'))
				if ax[row-6, col-6].is_first_col():
					ax[row-6, col-6].set_ylabel(features[row-6].replace(' ', '\n'))
		plt.legend(legend, loc='upper left', frameon=False, bbox_to_anchor=(1, 1))
		plt.show()
	except:
		print("error")
		exit_()

if __name__ == '__main__':
	try:
		dataset = load_csv('./datasets/dataset_train.csv')
		data = dataset[1:, :]
		data = data[data[:, 1].argsort()]
		features = dataset[0, 6:]
		legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
		pair_plot(data,features,legend)
		plt.show()
	except:
		print("error")
		exit_()