import numpy as np
from sklearn.neighbors import KNeighborsClassifier, VALID_METRICS
from KNN import KNN

# Columns that will use Hamming Distance (index starts at 0): 1, 3, 5, 6, 7, 8, 9, 13

weights_values = ['uniform', 'distance']
algorithm_values = ['ball_tree']

def dist(x, y):
    categorical_cols = [1, 3, 5, 6, 7, 8, 9]

    distance = 0
    for i in range(len(x)):
        if i not in categorical_cols:
            # Use Manhattan Distance
            distance = distance + np.absolute(x[i] - y[i])
        else:
            # Use Hamming Distance
            distance = distance + int((x[i] == y[i]) == True)

    return distance

result = ""
# print(sorted(VALID_METRICS['kd_tree']))
for k in range(100, 1000, 100):
    for weights in weights_values:
        for algorithm in algorithm_values:
            print("Start the training !")
            knn = KNN(total=1, ratio=0.8)
            model = knn.get_model(k, algorithm, weights, dist)
            errors = knn.get_loss(model=model)
            t_result = "Errors in KNN for k = {}, weight = {}, algorithm = {}: {}".format(k, weights, algorithm, errors)
            result = result + t_result + '\n'
            print("Finish KNN training and validation for k={}, weight={}, algorithm={}".format(k, weights, algorithm))

f = open('KNN_result.txt', 'w')
f.write(result)
f.close()
