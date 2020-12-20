#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import operator
from os import listdir
from collections import Counter


# In[2]:



def file2matrix(filename):
    fr = open(filename)
    numberlines = len(fr.readlines())
    returnMat = np.zeros((numberlines,13))
    classlabel = []#label
    fr = open(filename)
    index = 0
    for line in fr.readlines():
   
        listFromline = line.split(',')
        returnMat[index,:] = listFromline[0:13]
        classlabel.append(int(listFromline[-1]))
        index+=1
    return returnMat,classlabel


# In[3]:


file2matrix('trainingAdult.data')


# In[4]:


import matplotlib.pyplot as plt
import matplotlib

datingDataMat, datingLabels = file2matrix('trainingAdult.data')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*np.array(datingLabels), 15.0*np.array(datingLabels))
plt.show()


# In[5]:




def autoNorm(dataSet):
    
    minval = dataSet.min(0)
    maxval = dataSet.max(0)
    n = dataSet.shape
    normDataSet = np.zeros(n)
    m = dataSet.shape[0]
    ranges = maxval - minval

    normDataSet = dataSet - np.tile(minval,(m,1))
   
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    return normDataSet, ranges, minval


# In[6]:



def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
 
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
 
    sortedDistIndicies = distances.argsort()
   
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# In[16]:



def ClassTest(normalization):

    
    hoRatio = 0.1 
  
    datingDataMat, datingLabels = file2matrix('trainingAdult.data')  # load data setfrom file

    
    if normalization: normMat = datingDataMat
    else: normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]

    numTestVecs = int(m * hoRatio)
    print ('numTestVecs=', numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
     
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
#        print ("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): 
            errorCount += 1.0
#            print('error classify')
    print ("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print ("Error Count:", errorCount)


# In[17]:

print("Without Normalization")
ClassTest(True)
print("With Normalization")
ClassTest(False)


# In[ ]:




