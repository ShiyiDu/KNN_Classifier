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
    TESTING_FILE = 'testingAdult.test'
    data = []
    test_data = []
    training_X = []  # the features for training data
    training_y = []  # the result(class) for training data
    validation_X = []
    validation_y = []
    testing_X = []
    testing_y = []
    total = 0.5  # how much of the entire data we want to use?
    ratio = 0.8  # the how many data from training data are we using for the model?
    model = None
    testing = False

    def __init__(self, total=0.5, ratio=0.8, testing=False):
        self.data = load_data(self.TRAINING_FILE)
        if testing:
            self.testing = testing
            self.test_data = load_data(self.TESTING_FILE)
            # print("testing data:", len(self.test_data))
        self.ratio = ratio
        self.total = total
        self.re_sample()

    # re_sample the training and validation data
    def re_sample(self):
        random.shuffle(self.data)
        d_len = int(len(self.data) * self.total)
        t_len = int(self.ratio * d_len)
        training = self.data[0: t_len]
        validation = self.data[t_len: d_len]

        print("training set: ", len(training))
        print("validation set: ", len(validation))

        if self.testing:
            self.testing_X = np.array(list(map(lambda arr: arr[0:-1], self.test_data)))
            self.testing_y = np.array(list(map(lambda arr: arr[-1], self.test_data)))
            print("testing set: ", len(self.testing_y))

        self.training_X = np.array(list(map(lambda arr: arr[0:-1], training)))
        self.training_y = np.array(list(map(lambda arr: arr[-1], training)))

        self.validation_X = np.array(list(map(lambda arr: arr[0:-1], validation)))
        self.validation_y = np.array(list(map(lambda arr: arr[-1], validation)))

    def get_model(self, k, dist):
        X = self.training_X
        y = self.training_y
        # print(y[0:15])
        # my_dist = DistanceMetric.get_metric('pyfunc', func=dist)
        nbrs = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='ball_tree', metric='pyfunc',
                                    metric_params={'func': dist})
        nbrs.fit(X, y)
        # print(nbrs.kneighbors([[36, 3, 220696, 11, 9, 0, 6, 1, 4, 1, 0, 0, 40, 38]]))
        # print("prediction:", nbrs.predict([[52, 5, 240013, 12, 14, 2, 11, 0, 4, 1, 0, 0, 70, 38]]))
        self.model = nbrs
        return nbrs

    # calculate the 1-0 loss of the model from the validation data
    def get_loss(self, model=model, write_file=None):
        if model is None:
            return -1
        predictions = model.predict(self.validation_X if not self.testing else self.testing_X)
        total = len(predictions)
        errors = 0

        if not write_file is None:
            new_file = open(write_file, "w")
            content = "\n".join(map(str, predictions))
            new_file.write(content)
            new_file.close()

        if not self.testing:
            for i in range(total):
                if not predictions[i] == self.validation_y[i]:
                    errors += 1
        else:
            for i in range(total):
                if not predictions[i] == self.testing_y[i]:
                    errors += 1

        return errors / float(total)

# knn = KNN(total=0.1, ratio=0.8)
# model = knn.get_model(100, lambda x, y: np.sum((x - y) ** 2))
# knn.re_sample()
# print(knn.get_loss())
# knn.re_sample()
# print(knn.get_loss())
# knn.re_sample()
# print(knn.get_loss())
