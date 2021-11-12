import numpy as np
from skimage import io

dict_class_name_to_number = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,
                             7: 7, 8: 8, 9: 9}
dict_number_to_class_name = dict_class_name_to_number


def plot_digit_mnist(x_set, y_set, idx):
    img = x_set.T[idx].reshape(28, 28)
    io.imshow(img)
    print(y_set.T[idx])
    io.show()


def load_data():
    train_dataset = np.loadtxt('./digit-recognizer/' + 'train.csv', skiprows=1, delimiter=',')
    x_train_set, y_train_set = train_dataset[:4000, 1:], train_dataset[:4000, 0]
    x_test_set, y_test_set = train_dataset[4000:, 1:], train_dataset[4000:, 0]
    x_train_set, x_test_set = x_train_set.T / 255., x_test_set.T / 255.
    y_train_set, y_test_set = y_train_set.reshape((1, y_train_set.shape[0])), y_test_set.reshape(
        (1, y_test_set.shape[0]))
    return x_train_set, y_train_set, x_test_set, y_test_set


def load_data_submission():
    test_dataset = np.loadtxt('./digit-recognizer/' + 'test.csv', skiprows=1, delimiter=',')
    x_submission = test_dataset.T / 255.
    return x_submission


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    print(x_test.shape)
    print(y_test.shape)
