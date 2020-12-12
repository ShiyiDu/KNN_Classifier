import numpy as np
import matplotlib.pyplot as plt

workClass = np.full((8, 2), 0)
education = np.full((16, 2), 0)
maritalStatus = np.full((7, 2), 0)
occupation = np.full((14, 2), 0)
relationship = np.full((6, 2), 0)
race = np.full((5, 2), 0)
sex = np.full((2, 2), 0)
nativeCountry = np.full((41, 2), 0)

# def updateMarker(dictionary, field, target_value):
#     if target_value == 0:
#         dictionary.get(field)[0] = dictionary.get(field)[0] + 1;
#     else:
#         dictionary.get(field)[1] = dictionary.get(field)[1] + 1;


with open('trainingAdult.data') as content:
    for line in content:
        values = line.split(',')
        target_value = int(values[-1]);
        workClass[int(values[1]), target_value] += 1;
        education[int(values[3]), target_value] += 1;
        maritalStatus[int(values[5]), target_value] += 1;
        occupation[int(values[6]), target_value] += 1;
        relationship[int(values[7]), target_value] += 1;
        race[int(values[8]), target_value] += 1;
        sex[int(values[9]), target_value] += 1;
        nativeCountry[int(values[13]), target_value]+=1


def plot_field(data, labels):
    lte_than_50k = np.transpose(data[:, 0])
    gt_than_50k = np.transpose(data[:, 1])
    # total = np.asarray([v[0] + v[1] for v in data])

    ind=np.arange(len(labels))
    width = 0.3

    p1 = plt.barh(ind, lte_than_50k, width)
    p2 = plt.barh(ind, gt_than_50k, width, left=lte_than_50k)
    plt.yticks(ind, labels)
    plt.legend((p1[0], p2[0]), ('Less than or equal to 50k', "Greater then 50k"))
    plt.show()


workclassLabels = ["Federal-gov", "Local-gov", "Never-worked", "Private", "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"]
educationLabels = [
    "10th",
    "11th",
    "12th",
    "1st-4th",
    "5th-6th",
    "7th-8th",
    "9th",
    "Assoc-acdm",
    "Assoc-voc",
    "Bachelors",
    "Doctorate",
    "HS-grad",
    "Masters",
    "Preschool",
    "Prof-school",
    "Some-college"
]
# maritalstatusLabels =

occupationLabels = ["Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial", "Farming-fishing",
"Handlers-cleaners", "Machine-op-inspct", "Other-service", "Priv-house-serv",
"Prof-specialty", "Protective-serv", "Sales", "Tech-support", "Transport-moving"]

relationshipLabels = ["Husband", "Not-in-family", "Other-relative", "Own-child", "Unmarried", "Wife"]
raceLabels =["Amer-Indian-Eskimo", "Asian-Pac-Islander", "Black", "Other", "White"]
sexLabels = ["Female", "Male"]
nativeCountryLabels = [
    "Cambodia",
    "Canada",
    "China",
    "Columbia",
    "Cuba",
    "Dominican-Republic",
    "Ecuador",
    "El-Salvador",
    "England",
    "France",
    "Germany",
    "Greece",
    "Guatemala",
    "Haiti",
    "Holand-Netherlands",
    "Honduras",
    "Hong",
    "Hungary",
    "India",
    "Iran",
    "Ireland",
    "Italy",
    "Jamaica",
    "Japan",
    "Laos",
    "Mexico",
    "Nicaragua",
    "Outlying-US(Guam-USVI-etc)",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Puerto-Rico",
    "Scotland",
    "South",
    "Taiwan",
    "Thailand",
    "Trinadad&Tobago",
    "United-States",
    "Vietnam",
    "Yugoslavia"
]

# Run the plot
# plot_field(workClass, workclassLabels)
# plot_field(nativeCountry, nativeCountryLabels)
# plot_field(occupation, occupationLabels)
plot_field(education, educationLabels)
