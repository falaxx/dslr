import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from describe import describe,load_csv

def pair_plot_scatter(ax, x, y):
  ax.scatter(x[:327], y[:327], s=1, color='red', alpha=0.5)
  ax.scatter(x[327:856], y[327:856], s=1, color='yellow', alpha=0.5)
  ax.scatter(x[856:1299], y[856:1299], s=1, color='blue', alpha=0.5)
  ax.scatter(x[1299:], y[1299:], s=1, color='green', alpha=0.5)

def pair_plot(dataset,features,legend):

	_, ax = plt.subplots(nrows=13, ncols=13, sharex='col', sharey='row',gridspec_kw={'hspace': 0, 'wspace': 0})
	for row in range(6, 19):
		for col in range(6, 19):
			# un subplot par combinaison
			ax[row-6, col-6].tick_params(labelbottom=False)
			ax[row-6, col-6].tick_params(labelleft=False)
			x = np.array(dataset[:, col], dtype=float)
			y = np.array(dataset[:, row], dtype=float)
			pair_plot_scatter(ax[row-6,col-6], x, y)
			# mets le label  
			if ax[row-6, col-6].is_last_row():
				ax[row-6, col-6].set_xlabel(features[col-6].replace(' ', '\n'))
			if ax[row-6, col-6].is_first_col():
				ax[row-6, col-6].set_ylabel(features[row-6].replace(' ', '\n'))
	plt.legend(legend, loc='upper left', frameon=False, bbox_to_anchor=(1, 1))
	plt.show()

if __name__ == '__main__':
	dataset = load_csv('./datasets/dataset_train.csv')
	data = dataset[1:, :]
	data = data[data[:, 1].argsort()]
	features = dataset[0, 6:]
	legend = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']

	pair_plot(data,features,legend)
	plt.show()