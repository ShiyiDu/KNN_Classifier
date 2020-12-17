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
 
def handleDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'rb') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split: trainingSet.append(dataset[x]) else: testSet.append(dataset[x])
                    def euclideanDistance(instance1, instance2, length): distance = 0 for x in range(length): distance += pow((instance1[x] - instance2[x]), 2) 
                            return math.sqrt(distance) 
                        def getNeighbors(trainingSet, testInstance, k): distances = [] length = len(testInstance)-1 for x in range(len(trainingSet)): 
                                dist = euclideanDistance(testInstance, trainingSet[x], length) 
                                distances.append((trainingSet[x], dist)) 
                                distances.sort(key=operator.itemgetter(1)) 
                                neighbors = [] 
                                for x in range(k): neighbors.append(distances[x][0]) 
                                    return neighbors def getResponse(neighbors): classVotes = {}
                                    for x in range(len(neighbors)): response = neighbors[x][-1] 
                                        if response in classVotes: classVotes[response] += 1 
                                            else: classVotes[response] = 1 sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
                                                return sortedVotes[0][0] 
                                            def getAccuracy(testSet, predictions): 
                                                correct = 0 for x in range(len(testSet)):
                                                    if testSet[x][-1] == predictions[x]: correct += 1 
                                                        return (correct/float(len(testSet))) * 100.0 
                                                    def main(): # prepare data trainingSet=[] testSet=[] split = 0.67 loadDataset('iris.data', split, trainingSet, testSet) print 'Train set: ' + repr(len(trainingSet)) print 'Test set: ' + repr(len(testSet)) # generate predictions predictions=[] k = 3 for x in range(len(testSet)): neighbors = getNeighbors(trainingSet, testSet[x], k) result = getResponse(neighbors) predictions.append(result) print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')
     
main()


# In[11]:


TRAINING_FILE = 'trainingAdult.data'
data = []
training_X = []  # the features for training data
training_y = []  # the result(class) for training data
validation_X = []
validation_y = []
total = 0.5  # how much of the entire data we want to use?
ratio = 0.8  # the how many data from training data are we using for the model?
model = None

def __init__(self, total=0.5, ratio=0.8):
    self.data = load_data(self.TRAINING_FILE)
    self.ratio = ratio
    self.total = total
    self.re_sample()
    
def re_sample(self):
    random.shuffle(self.data)
    d_len = int(len(self.data) * self.total)
    t_len = int(self.ratio * d_len)
    training = self.data[0: t_len]
    validation = self.data[t_len: d_len]

    print("training set: ", len(training))
    print("validation set: ", len(validation))

    self.training_X = np.array(list(map(lambda arr: arr[0:-1], training)))
    self.training_y = np.array(list(map(lambda arr: arr[-1], training)))

    self.validation_X = np.array(list(map(lambda arr: arr[0:-1], validation)))
    self.validation_y = np.array(list(map(lambda arr: arr[-1], validation)))


 #   __init__   
 #   trainingSet=self.training_X
 #   testSet=self.validation_X 
 #   testInstance = []
  #  for i in range (len(testSet)):
 #       testInstance.append(1)

 #   neighbors = getKNeighbors(trainSet, testInstance, 1)


# In[ ]:




