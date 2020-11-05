import argparse
import csv
import numpy as np
import matplotlib
from decimal import Decimal

def mean(X):
	res = 0
	for i in range (0,len(X)):
		res += X[i]
	return round((res / len(X)),6)

def min(X):
	res = 1000000.0
	for i in range (0,len(X)):
		if X[i] < res:
			res = X[i]
	return round(res,6)

def max(X):
	res = -1000000.0
	for i in range (0,len(X)):
		if X[i] > res:
			res = X[i]
	return round(res,6)

def std(X):
	m = mean(X)
	res = 0
	for i in range (0,len(X)):
		res += (X[i] - m) ** 2
	res = (res / len(X)) ** 0.5
	return round(res,6)


def percentile(X, percent):
	X.sort()
	rank = percent * (len(X) - 1)
	f = np.floor(rank)
	c = np.ceil(rank)
	if f == c:
		return round(X[int(rank)],6)
	return round((X[int((c + f) / 2)]), 6)


def describe(filename):
	try:
		dataset = load_csv(filename)
		features = dataset[0]
		dataset = dataset[1:, :]
		string = "     |"
		for i in range(6,len(features)):
			features[i] = '%.12s' % features[i]
			string += (f'{features[i]:>12}|')
	except:
		print("rip")
	print(string)
	print(f'{"Count":<5}', end='|')
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{len(data):>12}', end='|')
	print()
	print(f'{"Mean":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{mean(data):>12}', end='|')
	print()
	print(f'{"Std":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{std(data):>12}', end='|')
	print()
	print(f'{"Min":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{min(data):>12}', end='|')
	print()
	print(f'{"25%":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{percentile(data,0.25):>12}', end='|')
	print()
	print(f'{"50%":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{percentile(data,0.5):>12}', end='|')
	print()
	print(f'{"75%":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{percentile(data,0.75):>12}', end='|')
	print()
	print(f'{"Max":<5}', end='|')	
	for i in range(6, len(features)):
		data = np.array(dataset[:, i], dtype=float)
		data = data[~np.isnan(data)]
		print(f'{max(data):>12}', end='|')
	print()
	# print(f'{"realmed":<5}', end='|')	
	# for i in range(6, len(features)):
	# 	data = np.array(dataset[:, i], dtype=float)
	# 	data = data[~np.isnan(data)]
	# 	print(f'{np.percentile(data,25):>12}', end='|')
	# for i in range(6, len(features)):
	# 	data = np.array(dataset[:, i], dtype=float)
	# 	data = data[~np.isnan(data)]
	# 	print(f'{len(data):>10}', end='|')
	
	# for i in range(6, len(features)):
	# 	data = np.array(dataset[:, i], dtype=float)
	# 	data = data[~np.isnan(data)]
	# 	print(f'{len(data):>10}', end='|')

	# for i in range(6, len(features)):
	# 	data = np.array(dataset[:, i], dtype=float)
	# 	data = data[~np.isnan(data)]
	# 	print(f'{len(data):>10}', end='|')
		# except:
		# 	print("rip")
	# print(f'{features[i]:15.15}', end=' |')
	# try:
	# 	data = np.array(dataset[:, i], dtype=float)
	# 	data = data[~np.isnan(data)]
	# 	if not data.any():
	# 	raise Exception()
	# 	print(f'{count_(data):>12.4f}', end=' |')
	# 	print(f'{mean_(data):>12.4f}', end=' |')
	# 	print(f'{std_(data):>12.4f}', end=' |')
	# 	print(f'{min_(data):>12.4f}', end=' |')
	# 	print(f'{percentile_(data, 25):>12.4f}', end=' |')
	# 	print(f'{percentile_(data, 50):>12.4f}', end=' |')
	# 	print(f'{percentile_(data, 75):>12.4f}', end=' |')
	# 	print(f'{max_(data):>12.4f}')

	# 	print(f'{count_(dataset[:, i]):>12.4f}', end=' |')
	# 	print(f'{"No numerical value to display":>60}')

def load_csv(filename):
	dataset = list()
	try:
		with open(filename) as csvfile:
			reader = csv.reader(csvfile)
			for _ in reader:
				row = list()
				for value in _:
					try:
						value = float(value)
					except:
						if not value:
							value = np.nan
					row.append(value)
				dataset.append(row)
		return np.array(dataset, dtype=object)
	except:
		print("error")



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset", type=str, help="input dataset")
	args = parser.parse_args()
	describe(args.dataset)
	

if __name__ == "__main__":
	main()