import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('newTrainingAdult.csv', header=0)
test_data = pd.read_csv('newTestingAdult.csv', header=0)

print("Size of the train data: {}".format(len(train_data.index)))
print("Size of the test data: {}".format(len(test_data.index)))

# print(train_data.head())
# print(test_data.head())
# print(train_data.columns)
# print(test_data.columns)

total_errors = 0
total_errors_test = 0
for i in range(5):
    print("Trial {}".format(i))
    temp_train_data = train_data.sample(frac=1).reset_index(drop=True)

    X = temp_train_data.drop('<=50k', axis=1)
    y = temp_train_data['<=50k']

    X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=30)

    # print(len(X_train))
    # print(y_validate.to_numpy())

    # p = 1: Manhattan Distance
    # p = 2 (Default): Euclidean Distance
    w = ['uniform', 'distance']
    ls = [30, 50, 100]
    k = [28, 500, 2000]

    model = KNeighborsClassifier(n_neighbors=k[1], algorithm='kd_tree', p=2, weights=w[1], leaf_size=ls[2])

    model.fit(X_train, y_train)
    # print("Finish the training")
    y_pred = model.predict(X_validate)
    # print("Finish the prediction")

    error = 0;
    total = len(y_pred)
    y_target = y_validate.to_numpy()

    for i in range(total):
        if not y_target[i] == y_pred[i]:
            error += 1

    error_percentage = error / total
    total_errors = total_errors + error_percentage

    print("Error in the validation data (20% of the train data): {}".format(error_percentage))
    # print("Start testing the model with the test data")

    X_test = test_data.drop('<=50k', axis=1)
    y_test = test_data['<=50k']

    y_test_pred = model.predict(X_test)

    error = 0
    total = len(y_test_pred)
    y_target = y_test.to_numpy()

    for i in range(total):
        if not y_target[i] == y_test_pred[i]:
            error += 1

    error_percentage = error / total
    total_errors_test = total_errors_test + error_percentage

    print("Error in the test data (20% of the total data): {}".format(error_percentage))

average_errors = total_errors / 5
average_errors_test = total_errors_test / 5
print("Average Errors in the validation data (20% of the train data): {}".format(average_errors))
print("Average Errors in the test data (20% of the total data): {}".format(average_errors_test))
