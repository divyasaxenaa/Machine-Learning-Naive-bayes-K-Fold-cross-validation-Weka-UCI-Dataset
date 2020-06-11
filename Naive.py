from csv import reader
from csv import writer
from csv import QUOTE_ALL
from random import seed
from random import randrange
from math import sqrt
from math import exp
from math import pi
import Graphplot as gr



def __render_row(row_value):
	dataset = list()
	for row in row_value:
		if row:
			dataset.append(row)
	return dataset


def __split_dataset(dataset, number_of_folds, fold_size):
	dataset_split = list()
	for _ in range(number_of_folds):
		fold_value = __make_fold(fold_size, dataset)
		dataset_split.append(fold_value)
	return dataset_split


def __make_fold(fold_size, dataset):
	fold = list()
	while len(fold) < fold_size:
		index = randrange(len(dataset))
		fold.append(dataset.pop(index))
	return fold


def __calculate_score(folds, algorithm, *args):
	scores = list()
	for fold in folds:
		train_set = __create_train_set(fold, folds)
		test_set = __create_test_set(fold)
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores


def __create_train_set(fold, folds):
	train_set = list(folds)
	train_set.remove(fold)
	return sum(train_set, [])


def __create_test_set(fold):
	test_set = list()
	for row in fold:
		row_copy = list(row)
		test_set.append(row_copy)
	return test_set


def load_csv(filename):
	with open(filename, 'r') as file:
		row_value = reader(file)
		return __render_row(row_value)



def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())


def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]

	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i

	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup


def cross_validation_split(dataset, number_of_folds):
	fold_size = int(len(dataset) / number_of_folds)
	return __split_dataset(dataset, number_of_folds, fold_size)


def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	return __calculate_score(folds, algorithm, *args)


def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated


def mean(numbers):
	return sum(numbers)/float(len(numbers))


def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)


def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries


def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries


def calculate_probability(x, mean, stdev):
	if stdev == 0 :
		stdev = 1
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent


def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
			probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
	return probabilities


def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label


def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)


def main(dataset_selected):
	seed(1)
	if dataset_selected == "car":
		filename = 'car.csv'
	elif dataset_selected == "hayes":
		filename = 'hayes-roth.csv'
	elif dataset_selected == "iris":
		filename = 'iris.csv'
	else:
		filename = 'breast-cancer.csv'
	dataset = load_csv(filename)
	if dataset_selected == "car":
		str_column_to_int(dataset, 0)
		str_column_to_int(dataset, 1)
		str_column_to_int(dataset, 2)
		str_column_to_int(dataset, 3)
		str_column_to_int(dataset, 4)
		str_column_to_int(dataset, 5)
		str_column_to_int(dataset, 6)
	elif dataset_selected == "hayes":
		for i in range(len(dataset[0]) - 1):
			str_column_to_float(dataset, i)
		str_column_to_int(dataset, len(dataset[0])-1)
	elif dataset_selected == "iris":
		for i in range(len(dataset[0]) - 1):
			str_column_to_float(dataset, i)
		# convert class column to integers
		str_column_to_int(dataset, len(dataset[0]) - 1)
	else:
		str_column_to_int(dataset, 0)
		str_column_to_int(dataset, 1)
		str_column_to_int(dataset, 2)
		str_column_to_int(dataset, 3)
		str_column_to_int(dataset, 4)
		str_column_to_int(dataset, 5)
		str_column_to_float(dataset, 6)
		str_column_to_int(dataset, 7)
		str_column_to_int(dataset, 8)
		str_column_to_int(dataset, 9)
	with open("data.csv", 'w', newline='') as myfile:
		 wr = writer(myfile, quoting = QUOTE_ALL, delimiter = '\n')
		 wr.writerow(dataset)
	gr.main(dataset_selected)
	n_folds = 10
	scores = evaluate_algorithm(dataset, naive_bayes, n_folds)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
