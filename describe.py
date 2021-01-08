import argparse
import csv
import numpy as np
import matplotlib
from utils import mean, min, max, std, percentile

def describe(filename):
	try:
		dataset = load_csv(filename)
		features = dataset[0]
		dataset = dataset[1:, :]
		string = "      "
		for i in range(6,len(features)):
			features[i] = '%.12s' % features[i]
			string += (f'{features[i]:>12}|')
	except:
		print("error")
		return
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
	return data

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
	parser.add_argument("dataset", type=str, help="dataset")
	args = parser.parse_args()
	describe(args.dataset)
	

if __name__ == "__main__":
	main()