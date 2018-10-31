import pickle, gzip, random, time
import numpy as np

import matplotlib.pyplot as plt
import pylab as pl

fl = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(fl, encoding="latin1")
fl.close()

# f = open("rn-"+ time.strftime("%Y-%m-%d-%H-%M-%S"),"w")
ITERATIONS = 10000
LEARNING_RATE = 0.04

class Perceptron:
	def __init__(self, digit, weights = [np.random.standard_normal() for i in range(784)], bias = np.random.standard_normal()):
		self.weights = weights
		self.bias = bias
		self.digit = digit

def activation(x):
	if x > 0:
		return 1
	return 0

def train(training_set, digit):
	global ITERATIONS
	global LEARNING_RATE
	iterations = ITERATIONS
	count = 0
	bias = np.random.standard_normal()
	w = np.array([np.random.standard_normal() for i in range(784)])
	all_classified = False
	while not all_classified and iterations > 0:
		all_classified = True
		for x,target in training_set:
			if target == digit:
				t = 1
			else:
				t = 0
			x = np.array(x)
			target = np.array(target)
			z = np.dot(x,w) + bias
			output = activation(z)
			w = np.add(w, np.multiply(x, LEARNING_RATE * (t-output)))
			bias = bias + (t-output)*LEARNING_RATE
			if t != output:
				all_classified = False
		iterations -= 1
	return Perceptron(digit, w, bias)

def perceptron_decision(img, perceptron):
	return np.dot(img, perceptron.weights) + perceptron.bias

def predict(x, perceptrons):
	sum = 0
	detections = []
	for i in range(10):
		detections.append(perceptron_decision(x, perceptrons[i]))
	return detections.index(max(detections))

def test(test_set, perceptrons, f=None):
	# print("_______________TEST______________", file=f)
	# print("Prediction, Actual result", file=f)
	correct_detected = 0
	all_cases = 0
	wrong = 0
	for digit, target in test_set:
		all_cases += 1
		predicted = predict(digit, perceptrons)
		if target == predicted:
			correct_detected += 1
		# else:
			# print(predicted, target, file=f)
	print("Learning rate ", LEARNING_RATE)
	print("Iterations ", ITERATIONS)	
	# print("Learning rate ", LEARNING_RATE, file=f)
	# print("Iterations ", ITERATIONS, file=f)	
	print("Accuracy: ", correct_detected * 100 / all_cases, "%")
	# print("Accuracy: ", correct_detected * 100 / all_cases, "%", file=f)
	return correct_detected * 100 / all_cases

def print_perceptron(f, perceptron):
	print("Perceptron " + str(perceptron.digit), file=f)
	print(perceptron.weights, file=f)
	print(perceptron.bias, file=f)
	# print("**************************")


def main():
	x,t = train_set
	perceptrons = {}

	# lrs = list(pl.frange(0.01, 1, 0.01))
	accs = []
	# for lr in lrs:

	# 	global LEARNING_RATE
	# 	LEARNING_RATE	 = lr
	# iterations = range(10000, 20000, 100)
	# for it in iterations:
	# 	global ITERATIONS
	# 	ITERATIONS = it

	running_time = time.clock()
	for i in range(0,10):
		# print("Training perceptron ", i)
		perceptrons[i] = train(zip(x, t), i)
		# print_perceptron(f, perceptrons[i])
	running_time = time.clock() - running_time
	print("Training " + str(len(t)) + " instances took " + str(running_time) + "s")
	# print("Training " + str(len(t)) + " instances took " + str(running_time) + "s", file=f)
	print("Testing...")
	tx,tt = test_set
	# fle = open("rn-testing-"+ time.strftime("%Y-%m-%d-%H-%M-%S"),"w")
	running_time = time.clock()
	acc = test(zip(tx, tt), perceptrons)
	accs.append(acc)
	running_time = time.clock() - running_time
	print("Testing " + str(len(tt)) + " instances took " + str(running_time) + "s")
	# print("Testing " + str(len(tt)) + " instances took " + str(running_time) + "s", file=f)
	# f.close()
	# # fle.close()
	# print(lrs)
	# print(accs)
	
	# plt.plot(iterations, accs)
	# plt.show()
	# print("max acc: {}".format(max(accs)))
	# print("it: {}".format(iterations[accs.index(max(accs))]))

main()