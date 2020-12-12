import random
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train_data_file = "trainingAdult.data"
test_data_file = "testingAdult.test"

train_data = []
test_data = []

dictWorkclass = {
    0 : "Federal-gov",
    1 : "Local-gov",
    2 : "Never-worked",
    3 : "Private",
    4 : "Self-emp-inc",
    5 : "Self-emp-not-inc",
    6 : "State-gov",
    7 : "Without-pay"
}


dictEducation = {
    0 : "10th",
    1 : "11th",
    2 : "12th",
    3 : "1st-4th",
    4 : "5th-6th",
    5 : "7th-8th",
    6 : "9th",
    7 : "Assoc-acdm",
    8 : "Assoc-voc",
    9 : "Bachelors",
    10 : "Doctorate",
    11 : "HS-grad",
    12 : "Masters",
    13 : "Preschool",
    14 : "Prof-school",
    15 : "Some-college"
}

dictMaritalStatus = {
    0 : "Divorced",
    1 : "Married-AF-spouse",
    2 : "Married-civ-spouse",
    3 : "Married-spouse-absent",
    4 : "Never-married",
    5 : "Separated",
    6 : "Widowed"
}

dictOccupation = {
    0 : "Adm-clerical",
    1 : "Armed-Forces",
    2 : "Craft-repair",
    3 : "Exec-managerial",
    4 : "Farming-fishing",
    5 : "Handlers-cleaners",
    6 : "Machine-op-inspct",
    7 : "Other-service",
    8 : "Priv-house-serv",
    9 : "Prof-specialty",
    10 : "Protective-serv",
    11 : "Sales",
    12 : "Tech-support",
    13 : "Transport-moving"
}

dictRelationship = {
    0 : "Husband",
    1 : "Not-in-family",
    2 : "Other-relative",
    3 : "Own-child",
    4 : "Unmarried",
    5 : "Wife"
}

dictRace = {
    0 : "Amer-Indian-Eskimo",
    1 : "Asian-Pac-Islander",
    2 : "Black",
    3 : "Other",
    4 : "White"
}

dictNativecountry = {
    0 : "Cambodia",
    1 : "Canada",
    2 : "China",
    3 : "Columbia",
    4 : "Cuba",
    5 : "Dominican-Republic",
    6 : "Ecuador",
    7 : "El-Salvador",
    8 : "England",
    9 : "France",
    10 : "Germany",
    11 : "Greece",
    12 : "Guatemala",
    13 : "Haiti",
    14 : "Holand-Netherlands",
    15 : "Honduras",
    16 : "Hong",
    17 : "Hungary",
    18 : "India",
    19 : "Iran",
    20 : "Ireland",
    21 : "Italy",
    22 : "Jamaica",
    23 : "Japan",
    24 : "Laos",
    25 : "Mexico",
    26 : "Nicaragua",
    27 : "Outlying-US(Guam-USVI-etc)",
    28 : "Peru",
    29 : "Philippines",
    30 : "Poland",
    31 : "Portugal",
    32 : "Puerto-Rico",
    33 : "Scotland",
    34 : "South",
    35 : "Taiwan",
    36 : "Thailand",
    37 : "Trinadad&Tobago",
    38 : "United-States",
    39 : "Vietnam",
    40 : "Yugoslavia"
}
def revertLabelEncoder(data):
    data = data.astype({
        'workclass': 'string',
        'education': 'string',
        'marital-status': 'string',
        'occupation': 'string',
        'relationship': 'string',
        'race': 'string',
        'native-country': 'string',
    })

    for i in range(len(data.index)):
        data.at[i, 'workclass'] = dictWorkclass.get(int(data.at[i, 'workclass']))
        data.at[i, 'education'] = dictEducation.get(int(data.at[i, 'education']))
        data.at[i, 'marital-status'] = dictMaritalStatus.get(int(data.at[i, 'marital-status']))
        data.at[i, 'occupation'] = dictOccupation.get(int(data.at[i, 'occupation']))
        data.at[i, 'relationship'] = dictRelationship.get(int(data.at[i, 'relationship']))
        data.at[i, 'race'] = dictRace.get(int(data.at[i, 'race']))
        data.at[i, 'native-country'] = dictNativecountry.get(int(data.at[i, 'native-country']))

    data = data.astype({
        'workclass': 'string',
        'education': 'string',
        'marital-status': 'string',
        'occupation': 'string',
        'relationship': 'string',
        'race': 'string',
        'native-country': 'string',
    })

    return data

with open(train_data_file) as tad:
    for line in tad:
        values = line.split(',')
        temp_list = []
        for value in values:
            temp_list.append(int(value));
        train_data.append(temp_list);

with open(test_data_file) as ted:
    for line in ted:
        values = line.split(',')
        temp_list = []
        for value in values:
            temp_list.append(int(value));
        test_data.append(temp_list)

train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week","native-country", "<=50k"]
train_data = pd.DataFrame(data=train_data, columns=cols)
test_data = pd.DataFrame(data=test_data, columns=cols)
print("Finish preparing the data")

train_data = revertLabelEncoder(train_data)
test_data = revertLabelEncoder(test_data)
print("Finish reverting the label encoder")

X = train_data.drop("<=50k", axis=1)
y = train_data["<=50k"]

X_test = test_data.drop("<=50k", axis=1)
y_test = test_data["<=50k"]

irrelevant_features = ["native-country"]
categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race"]

X = X.drop(irrelevant_features, axis=1)
X_test = X_test.drop(irrelevant_features, axis=1)

# encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
#
# # One Hot Encoding the categorical_features
# train_cols = pd.DataFrame(encoder.fit_transform(X[categorical_features]))
# test_cols = pd.DataFrame(encoder.transform(X_test[categorical_features]))
# # Put the index back
# train_cols.index = X.index
# test_cols.index = X_test.index
#
# # Drop the old categorical columns
# num_X = X.drop(categorical_features, axis=1)
# num_X_test = X_test.drop(categorical_features, axis=1)
#
# # Merger the encoding with the remainders of data.
# encoded_X = pd.concat([num_X, train_cols], axis=1)
# encoded_X_test = pd.concat([num_X_test, test_cols], axis=1)
#
# new_train_data = pd.concat([encoded_X, y], axis=1)
# new_test_data = pd.concat([encoded_X_test, y_test], axis=1)
#
# print(new_train_data.head())

# https://stackoverflow.com/questions/37292872/how-can-i-one-hot-encode-in-python
def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[feature_to_encode])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res

for feature in categorical_features:
    X = encode_and_bind(X, feature)
    X_test = encode_and_bind(X_test, feature)

new_train_data = pd.concat([X, y], axis=1)
new_test_data = pd.concat([X_test, y_test], axis=1)

new_train_data.to_csv("newTrainingAdult.csv", index=False)
new_test_data.to_csv("newTestingAdult.csv", index=False)
