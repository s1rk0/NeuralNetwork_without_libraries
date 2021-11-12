import numpy as np

from NN import NNClassifier, NeuralNetwork, accuracy, ReLU, Regularization, plot_error
from load_cifar10 import load_data, dict_class_name_to_number, dict_number_to_class_name

x_train, y_train, x_test, y_test = load_data()

nn = NeuralNetwork(np.array([3072, 256, 64, 32, 10]), 0.1, activation=ReLU())
classifier = NNClassifier(nn, 2000)

classifier.fit(x_train, y_train, dict_class_name_to_number)

pred_train = classifier.predict(x_train, dict_number_to_class_name)
print('train set accuracy: ', accuracy(pred_train, y_train))
pred_test = classifier.predict(x_test, dict_number_to_class_name)
print('test set accuracy: ', accuracy(pred_test, y_test))

plot_error(classifier)
