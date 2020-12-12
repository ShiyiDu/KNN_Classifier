import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('newTrainingAdult.csv', header=0)
test_data = pd.read_csv('newTestingAdult.csv', header=0)

print("Size of the train data: {}".format(len(train_data.index)))
print("Size of the test data: {}".format(len(test_data.index)))
X = train_data.drop('<=50k', axis=1)
y = train_data['<=50k']

X_train, X_validate, y_train, y_validate = train_test_split(X, y, test_size=0.2, random_state=30)

# print(len(X_train))
# print(y_validate.to_numpy())

model = KNeighborsClassifier(n_neighbors=500, algorithm='kd_tree', p=1)

model.fit(X_train, y_train)
print("Finish the training")
y_pred = model.predict(X_validate)
print("Finish the prediction")

error = 0;
total = len(y_pred)
y_target = y_validate.to_numpy()

for i in range(total):
    if not y_target[i] == y_pred[i]:
        error += 1

error_percentage = error / total

print("Error in the validation data (20% of the train data): {}".format(error_percentage))
print("Start testing the model with the test data")

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

print("Error in the test data (20% of the total data): {}".format(error_percentage))
