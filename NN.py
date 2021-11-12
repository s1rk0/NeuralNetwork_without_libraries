import math

import numpy as np


def plot_error(model):
    import matplotlib.pyplot as plt
    plt.plot(range(len(model._cost)), model._cost)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.show()


def accuracy(pred, labels):
    return (np.sum(pred == labels, axis=1) / float(labels.shape[1]))[0]


def one_hot(y, dict_class_name_to_number, n_classes=10):
    """
    Encode labels into a one-hot representation
    :param y: array of input labels of shape (1, n_samples)
    :param n_classes: number of classes
    :param dict_class_name_to_number - dictionary class name - its number
    :return: onehot, a matrix of labels by samples. For each column, the ith index will be
    "hot", or 1, to represent that index being the label; shape - (n_classes, n_samples)
    """

    # Creating an array of class numbers from their labels
    arr = []
    for i in y[0, :]:
        arr.append(dict_class_name_to_number[i])
    arr = np.array(arr)

    #  Encode numbers of classes into a one-hot representation
    b = np.zeros((y.size, n_classes))
    b[np.arange(y.size), arr] = 1
    return b.T


class ReLU:
    def __call__(self, z):
        """
        Compute the sigmoid of z
        :param z: scalar or numpy array of any size
        :return: ReLU(z)
        """
        return z * (z > 0)

    def __str__(self):
        return 'ReLU'

    def prime(self, z):
        """
        Compute the derivative of ReLU of z
        :param z: scalar or numpy array of any size
        :return: ReLU prime
        """
        return 1 * (z > 0)

    def cost_function(self, last_a, y):
        m = y.shape[1]

        cost = 1 / m * np.sum((last_a - y) ** 2)

        return cost


class Sigmoid:
    def __call__(self, z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- scalar or numpy array of any size.

        Return:
        sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

    def __str__(self):
        return 'Sigmoid'

    def prime(self, z):
        """
        Compute the derivative of sigmoid of z

        Arguments:
        z -- scalar or numpy array of any size.

        Return:
        Sigmoid prime
        """
        return Sigmoid()(z) * (1 - Sigmoid()(z))

    def cost_function(self, last_a, y):
        m = y.shape[1]

        cost = - 1 / m * np.sum(y * np.log(last_a) + (1 - y) * np.log(1 - last_a))

        return cost


class Regularization:
    """
    Regularization class
    """

    def __init__(self, lambda_1, lambda_2):
        """
        :param lambda_1: regularization coeficient for l1 regularization
        :param lambda_2: regularization coeficient for l2 regularization
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def __str__(self):
        return '{0}'.format({'lambda_1': self.lambda_1, 'lambda_2': self.lambda_2})

    def l1(self, w, m):
        """
        Compute l1 regularization part
        :param w: list of weights (n_hidden_layers)
        :param m: n_examples
        :return: l1: float
        """
        sum_norm = 0
        for i in range(len(w)):
            sum_norm += np.linalg.norm(w[i], 1)
        return (self.lambda_1 / m) * sum_norm

    def l1_grad(self, w, m):
        """
        Compute l1 regularization term
        :param w: dict of weights (n_hidden_layers)
        :param m: n_examples
        :return: dict of l1_grads dw which are grads by corresponding weights
        """
        dw = {}
        for i in w:
            dw[i] = self.lambda_1 * np.sign(w[i]) / m
        return dw

    def l2(self, w, m):
        """
        Compute l2 regularization term
        :param w: list of weights (n_hidden_layers)
        :param m: n_examples
        :return: l2: float
        """
        sum_norm = 0
        for i in range(len(w)):
            sum_norm += np.linalg.norm(w[i]) ** 2
        return (self.lambda_2 / 2 / m) * sum_norm

    def l2_grad(self, w, m):
        """
        Compute l2 regularization term
        :param w: dict of weights (n_hidden_layers)
        :param m: n_examples
        :return: dict with l2_grads dw which are grads by corresponding weights
        """
        dw = {}
        for i in w:
            dw[i] = self.lambda_2 * w[i] / m
        return dw

    def elastic(self, w, m):
        """
        Compute elastic net regularization term
        :param w: dict of weights (n_hidden_layers)
        :param m: n_examples
        :return: dict with elastic dw which are grads by corresponding weights
        """
        dw = {}
        for i in w:
            dw[i] = self.l1_grad(w, m)[i] + self.l2_grad(w, m)[i]
        return dw


class NeuralNetwork:
    def __init__(self, n_units, learning_rate, reg=Regularization(0.1, 0.2),
                 activation=Sigmoid(), last_layer_activation=Sigmoid()):
        """
        :param n_features: int -- Number of features
        :param n_units: Number of units vector of shape (n_layers)
        :param learning_rate: float
        :param reg: instance of Regularization class
        :param activation: activation function
        :param last_layer_activation: activation function of last layer
        """
        self.learning_rate = learning_rate
        self.n_units = n_units
        self.reg = reg

        self.n_layers = self.n_units.size - 1

        self.activation = {self.n_layers: last_layer_activation}
        for i in range(1, self.n_layers):
            self.activation[i] = activation

        self.n_output_units = n_units[self.n_layers]
        self.w = None
        self.b = None

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        self.w: list of arrays of weights
        self.b: list of bios vectors
        :return: None
        """
        w = {}
        b = {}

        for i in range(self.n_layers):
            w[i] = 0.01 * np.random.randn(self.n_units[i + 1], self.n_units[i])
            b[i] = np.zeros((self.n_units[i + 1], 1))

        self.w = w
        self.b = b

    def __str__(self):
        return '{0}\n{1}\n{2}\n{3}\n{4}'.format(self.n_units, self.reg, self.activation, self.w, self.b)

    def forward_propagation(self, x):
        """
        :param x: input data of shape (number of features, number of examples)
        :return: dictionary containing lists 'z' and 'a'
        """
        z = {}  # Словарь входящих сигналов в каждый слой неронной сети
        a = {0: x}  # Словарь исходящих сигналов из каждого слоя неронной сети
        for i in range(self.n_layers):
            z[i + 1] = self.w[i] @ a[i] + self.b[i]
            a[i + 1] = self.activation[i + 1](z[i + 1])
        return {
            'z': z,
            'a': a
        }

    def backward_propagation(self, y, cache):
        m = cache['a'][0].shape[1]
        # Вычисляем ошибку выходного слоя и градиенты для матрицы весов и вектора смещения перед ним
        dz = {}
        dw = {}
        db = {}

        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                dz[i] = (cache['a'][self.n_layers] - y) * self.activation[i].prime(cache['z'][i])
            else:
                dz[i] = self.w[i].T @ dz[i + 1] * self.activation[i].prime(cache['z'][i])
            dw[i - 1] = (1 / m) * dz[i] @ cache['a'][i - 1].T + self.reg.elastic(self.w, m)[i - 1]
            db[i - 1] = (1 / m) * np.sum(dz[i], axis=1, keepdims=True)
        return {
            'dz': dz,
            'db': db,
            'dw': dw
        }

    def update_parameters(self, grads):
        for i in range(len(self.w)):
            self.w[i] -= self.learning_rate * grads['dw'][i]
            self.b[i] -= self.learning_rate * grads['db'][i]


class NNRegression:
    """
    NNClassifier class

    Arguments:
    model -- instance of NN
    epochs: int -- Number of epochs
    """

    def __init__(self, model, epochs=1000):
        self.model = model
        self.epochs = epochs
        self._cost = []  # Write value of cost function after each epoch to build graph later

    def fit(self, x, y):
        """
        Learn weights and errors from training data

        Arguments:
        X -- input data of shape (number of features, number of examples)
        Y -- labels of shape (1, number of examples)
        """

        self.model.initialize_parameters()
        for i in range(self.epochs):
            cache = self.model.forward_propagation(x)
            grads = self.model.backward_propagation(x, y, cache)
            self.model.update_parameters(grads)
            if i % 10 == 0:
                print(self.model.activation.cost_function(cache['a'][self.model.n_hidden_layers], y))
            self._cost.append(self.model.activation.cost_function(cache['a'][self.model.n_hidden_layers], y))

    def predict(self, x):
        """
        Generate array of predicted labels for the input dataset

        Arguments:
        X -- input data of shape (number of features, number of examples)

        Returns:
        predicted value
        """
        cache = self.model.forward_propagation(x)
        return cache['a'][self.model.n_hidden_layers]


class NNClassifier:
    """
    NNClassifier class
    """

    def __init__(self, model, epochs=1000):
        """
        :param model: instance of NN
        :param epochs: Number of epochs
        """
        self.model = model
        self.epochs = epochs
        self._cost = []  # Write value of cost function after each epoch to build graph later

    def fit(self, x, y, dict_class_name_to_number, batch_size=0, reset=True):
        """
        Learn weights and errors from training data
        :param x: input data of shape (number of features, number of examples)
        :param y: labels of shape (1, number of examples)
        :param batch_size: batch size int
        :param dict_class_name_to_number: dictionary class name - its number
        :param reset: reset weights or not?
        :return: None
        """
        if batch_size == 0:
            batch_size = x.shape[1]
        if reset:
            self.model.initialize_parameters()
            self._cost = []
        import time
        start_time = time.time()
        y = one_hot(y, dict_class_name_to_number, self.model.n_output_units)
        for i in range(self.epochs):
            for j in range(math.ceil(x.shape[1] / batch_size)):
                cache = self.model.forward_propagation(x[:, batch_size * j:batch_size * (j + 1)])
                grads = self.model.backward_propagation(y[:, batch_size * j:batch_size * (j + 1)], cache)
                self.model.update_parameters(grads)
            print('\rEpoch: {0}\tScore: {1}\tTime: {2}'.format(i,
                                                               self.model.activation[1].cost_function(
                                                                   self.model.forward_propagation(x)['a'][
                                                                       self.model.n_layers], y),
                                                               time.time() - start_time), end='')
            self._cost.append(self.model.activation[1].cost_function(
                self.model.forward_propagation(x)['a'][self.model.n_layers], y))
        print('')

    def predict(self, x, dict_number_to_class_name):
        """
        Generate array of predicted labels for the input dataset

        Arguments:
        X -- input data of shape (number of features, number of examples)

        Returns:
        predicted labels of shape (1, n_samples)
        """

        cache = self.model.forward_propagation(x)
        result = []
        for i in np.argmax(cache['a'][self.model.n_layers], axis=0).T:
            result.append(dict_number_to_class_name[i])
        return result

    def save_model(self, file_name="nn.txt"):
        f = open(file_name, 'w')
        f.write(str(self.model))

    # def load_model(self, file_name="nn.txt"):
    #     with open(file_name, 'r') as file:
    #         l = file.read()
    #         print(type(l))


if __name__ == "__main__":
    dict_class_name_to_number_mn = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                                    7: 7, 8: 8, 9: 9}
    dict_number_to_class_name_mn = dict_class_name_to_number_mn
    nn = NeuralNetwork(np.array([2, 4, 3]), 0.01)
    classifier = NNClassifier(nn, 120)
    # classifier.load_model()
