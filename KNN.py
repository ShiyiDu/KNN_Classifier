import random

import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# a static method for reading data from the file
def load_data(file_name):
    result = []
    # counter = 0
    with open(file_name) as content:
        for line in content:
            # counter += 1;
            # if (counter > 15):
            # return result;
            point = line.split(',')
            point = list(map(int, point))
            # print(point)
            if len(point) == 15:
                result.append(point)

    return result


class KNN:
    TRAINING_FILE = 'trainingAdult.data'
    data = []
    training_X = []  # the features for training data
    training_y = []  # the result(class) for training data
    validation_X = []
    validation_y = []
    ratio = 0.8  # the how many data from training data are we using for the model?
    model = None

    def __init__(self, ratio=0.8):
        self.data = load_data(self.TRAINING_FILE)
        self.re_sample()

    # re_sample the training and validation data
    def re_sample(self):
        random.shuffle(self.data)
        d_len = len(self.data)
        t_len = int(0.8 * d_len)
        training = self.data[0: t_len]
        validation = self.data[t_len: d_len]

        print("training set: ", len(training))
        print("validation set: ", len(validation))

        self.training_X = np.array(list(map(lambda arr: arr[0:-1], training)))
        self.training_y = np.array(list(map(lambda arr: arr[-1], training)))

        self.validation_X = np.array(list(map(lambda arr: arr[0:-1], validation)))
        self.validation_y = np.array(list(map(lambda arr: arr[-1], validation)))

    def get_model(self, k, dist, ratio):
        X = self.training_X
        y = self.training_y
        # print(y[0:15])
        # my_dist = DistanceMetric.get_metric('pyfunc', func=dist)
        nbrs = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', metric='pyfunc', metric_params={'func': dist})
        nbrs.fit(X, y)
        # print(nbrs.kneighbors([[36, 3, 220696, 11, 9, 0, 6, 1, 4, 1, 0, 0, 40, 38]]))
        # print("prediction:", nbrs.predict([[52, 5, 240013, 12, 14, 2, 11, 0, 4, 1, 0, 0, 70, 38]]))
        self.model = nbrs;
        return nbrs

    # calculate the 1-0 loss of the model from the validation data
    def get_loss(self):
        if self.model is None:
            return -1
        predictions = self.model.predict(self.validation_X)
        total = len(predictions)
        errors = 0
        for i in range(total):
            if not predictions[i] == self.validation_y[i]:
                errors += 1

        return errors / float(total)


knn = KNN(0.8)
model = knn.get_model(2, lambda x, y: np.sum((x - y) ** 2), 0.5)
print(knn.get_loss())
