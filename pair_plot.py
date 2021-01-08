import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from describe import describe,load_csv

def pair_plot_scatter(ax, X, y):
  ax.scatter(X[:327], y[:327], s=1, color='red', alpha=0.5)
  ax.scatter(X[327:856], y[327:856], s=1, color='yellow', alpha=0.5)
  ax.scatter(X[856:1299], y[856:1299], s=1, color='blue', alpha=0.5)
  ax.scatter(X[1299:], y[1299:], s=1, color='green', alpha=0.5)

def pair_plot(dataset, features, legend):
	font = {'family' : 'DejaVu Sans',
			'weight' : 'light',
			'size'   : 7}
	matplotlib.rc('font', **font)

	size = dataset.shape[1]
	_, ax = plt.subplots(nrows=size, ncols=size)
	plt.subplots_adjust(wspace=0.5, hspace=0.5,left=0, right=0.5,top=0.5, bottom=0)

	for row in range(0, size):
		for col in range(0, size):
			X = dataset[:, col]
			y = dataset[:, row]

		if col == row:
			pair_plot_scatter(ax[row, col], X, y)
		else:
			pair_plot_scatter(ax[row, col], X, y)

		if ax[row, col].is_last_row():
			ax[row, col].set_xlabel(features[col].replace(' ', '\n'))
		else:
			ax[row, col].tick_params(labelbottom=False)

		if ax[row, col].is_first_col():
			ax[row, col].set_ylabel(features[row].replace(' ', '\n'))
		else:
		# 	ax[row, col].tick_params(labelleft=False)
			ax[row, col].spines['right'].set_visible(False)
			ax[row, col].spines['top'].set_visible(False)

	plt.legend(legend, loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
	plt.show()

if __name__ == '__main__':
	dataset = load_csv('./datasets/dataset_train.csv')
	data = dataset[1:, 6:]
	data = data[data[:, ].argsort()]
	# print(data)
	features = dataset[0, 6:]
	legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	pair_plot(np.array(data, dtype=float), features, legend)
	plt.show()