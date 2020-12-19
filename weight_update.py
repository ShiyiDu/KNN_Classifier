#!/usr/bin/env python
# coding: utf-8

# In[20]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:



load_data('trainingAdult.data')


# In[4]:


import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

data1 = [2, 2, 2, 'a']
data2 = [4, 4, 4, 'b']
distance = euclideanDistance(data1, data2, 3)
print ('Distance: ' + repr(distance))


# In[8]:


import operator 
def getKNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1
neighbors = getKNeighbors(trainSet, testInstance, k)
print(neighbors)


# In[2]:


import operator
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
print(getResponse(neighbors))


# In[ ]:


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)


# In[ ]:


import csv
import random
import math
import operator
 




# In[11]:


TRAINING_FILE = 'trainingAdult.data'
data = []


# In[44]:


print('trainingAdult.test')

with open('testingAdult.test') as file:
    test_data = [[float(digit) for digit in line.split(',')] for line in file]

#print (test_data)

test_instance =[]
#for i in range(len(test_data[0])-1):
#    test_instance.append(5);
print(len(test_data[0])-1)
test_instance = [3, 9, 0, 0, 9, 4, 2, 8, 5, 6, 6, 9, 2, 9]
print(len(test_instance))

print(test_instance)

k=3
neighbors = getKNeighbors(test_data, test_instance, k)
print(neighbors)

print(getResponse(neighbors))

neighbors = getKNeighbors(test_data, test_instance, k)
print(neighbors)

print(getResponse(neighbors))

import pandas as pd
df = pd.read_csv('testingAdult.test', sep=',', header=None )
predict_arr = df[df.columns[-1]].to_numpy()
print("predict_arr")
print(predict_arr) 

test_arr = df[df.columns[0:len(df)-2]].to_numpy()
#test_arr

k=3
neighbors = getKNeighbors(test_arr, test_instance, k)
print(neighbors)

print(getResponse(neighbors))


# In[26]:



testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)

