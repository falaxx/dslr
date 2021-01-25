import argparse
import csv
import numpy as np
import matplotlib
from utils import mean_, min_, max_, std_, percentile_, load_csv

def describe():
	try:
		dataset = load_csv('./datasets/dataset_train.csv')
		features = dataset[0]
		dataset = dataset[1:, :]
		string = "      "
		for i in range(6,len(features)):
			features[i] = '%.12s' % features[i]
			string += (f'{features[i]:>12}|')
		print(string)
		print(f'{"Count":<5}', end='|')
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{len(data):>12}', end='|')
		print()
		print(f'{"mean":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{mean_(data):>12}', end='|')
		print()
		print(f'{"std":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{std_(data):>12}', end='|')
		print()
		print(f'{"min":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{min_(data):>12}', end='|')
		print()
		print(f'{"25%":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{percentile_(data,0.25):>12}', end='|')
		print()
		print(f'{"50%":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{percentile_(data,0.5):>12}', end='|')
		print()
		print(f'{"75%":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{percentile_(data,0.75):>12}', end='|')
		print()
		print(f'{"max":<5}', end='|')	
		for i in range(6, len(features)):
			data = np.array(dataset[:, i], dtype=float)
			data = data[~np.isnan(data)]
			print(f'{max_(data):>12}', end='|')
		print()
		return data
	except:
		print("error")
		exit(0)
		return


def main():
	describe()
	

if __name__ == "__main__":
	main()