# -*- coding: utf-8 -*-
import numpy as np
import random
from ISBN9787111597674.chapter10 import mnist_loader


class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def backprop(self, x, y):
        """return a tuple
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]

        zs = []

        # forward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)

        # backprop
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
            return the number of test inputs for which is correct
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)

    def feedforward(self, a):
        """
            Return the output of the network if "a " is input
        """
        for b, w in zip(self.biases, self.weights):
            a = self.sigmoid(np.dot(w, a) + b)

        return a

    def update_mini_batch(self, mini_batch, eta):
        """
            update the networks' weights and biases by applying gradient descent using
            bp to a single mini batch
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) *
                        nw for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b - (eta / len(mini_batch)) *
                       nb for b, nb in zip(self.biases, nabla_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic
        gradient descent, the "training_data" is a list of tuples
        (x,y) representing the training inputs and the desired outputs.
        the other non-optional params are self-explanatory
        """
        if test_data:
            n_test = len(test_data)

        n = len(training_data)  # 50000
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {0}:{1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


if __name__ == '__main__':
    net = Network([784, 100, 10])
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
