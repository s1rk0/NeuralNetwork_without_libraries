import numpy as np
from skimage import io

dict_class_name_to_number = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6,
                             'horse': 7, 'ship': 8, 'truck': 9}
dict_number_to_class_name = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                             7: 'horse', 8: 'ship', 9: 'truck'}



def plot_image(x_set, y_set, idx):
    img = x_set.T[idx].reshape(32, 32, 3)
    io.imshow(img)
    print(y_set.T[idx])
    io.show()


def load_data():
    import pandas as pd
    labels_df = pd.read_csv('./input/' + 'trainLabels.csv', index_col='id')
    images = []
    for i in range(1, 50001):
        if i % 100 == 0:
            print('\rData loading {0}%'.format(i / 50000 * 100), end='')
        images.append(io.imread('./input/train/' + '{0}.png'.format(i)).flatten())
    print('')
    images_array = np.array(images).T / 255.
    labels_array = np.array(labels_df.values).T
    x_train, y_train = images_array[:, :5000], labels_array[:, :5000]
    x_test, y_test = images_array[:, 5000:], labels_array[:, 5000:]
    return x_train, y_train, x_test, y_test


def load_data_submission():
    images = []
    for i in range(1, 300001):
        if i % 100 == 0:
            print('Data for submission loading {0}%'.format(i / 300000 * 100))
        images.append(io.imread('./input/test/' + '{0}.png'.format(i)).flatten())
    x_submission = np.array(images).T / 255.
    return x_submission


if __name__ == "__main__":
    x_train, y_train, test_x, test_y = load_data()
    print(x_train.shape)
