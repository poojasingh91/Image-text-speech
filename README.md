# Image-text-speech

# Extracted google voice using gTTS library

from gtts import gTTS
import os
mytext='Hello world! This is a sample example of our Major project'
language='en'
myobj=gTTS(text=mytext,lang=language,slow=False)
myobj.save("welcome.mp3")
os.system("mpg321 welcome.mp3")


# 20,000 datasets for alphabets: 10,000 for training and 10,000 for testing.Splitting data for features and responses
import cv2 as cv
import numpy as np
# Load the data, converters converts the letter to a number
data= np.loadtxt('letter-recognition.data', dtype='float32',delimiter=',', converters={0:lambda ch:ord(ch)-ord('A')})
# Split the data to two, 10,000 each for train and test
train, test = np.vsplit(data,2)
# Split trainData and testData to features and resposes
responses, trainData = np.hsplit(train,[1])
labels, testData = np.hsplit(test,[1])
# Initiate the kNN, classify, measure accuracy.
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)
correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000

# 5000 hand written digits each sized 20×20
# 500 different numbers for each digit
# 2500 for training and 2500 for testing

import numpy as np
import cv2 as cv
img = cv.imread('digits.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# Now we split the image to seee cells, each 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
# Make it into a Numpy array. It size wilt be (50,108,28,20)
x = np.array(cells)
# Now we prepare train_data and test_data. 
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2588,488) 
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)
          
# Create Labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_labels = train_labels.copy()
# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()
knn.train(train, cv.ml.ROW_SAMPLE, train_labels)   ##Error
ret,result,neighbours,dist = knn.findNearest(test,k=5)
# Now we check the accuracy of classification
# For that, compare the result with test_Labels and check which are wrong
matches = result==test_labels
correct = np. count_nonzero(matches)
accuracy = correct*100.0/result.size
print("The accuracy of the digits is: " + str(accuracy) + "%." )

# Increasing accuracy by reducing overfitting
# Feeding training datasets with error data
# Saving obtained data to training set for each iteration

# save the data
np.savez('knn_data.npz',train=train, train_labels=train_labels)
# Now Load the data
with np.load('knn_data.npz') as data:
    print(data.files)
    train = data['train']
    train_labels = data['train_labels']
print("The accuracy of the letters is: " + str(accuracy) + ".")
