#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import tensorflow as tf
import pandas as pd 
from keras.utils.np_utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,MaxPool2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import confusion_matrix,accuracy_score


# In[83]:


## Read in Dataset
dataTrain = pd.read_csv("sign_mnist_train.csv")
dataTest = pd.read_csv("sign_mnist_test.csv")

yTrain, yTest = dataTrain["label"], dataTest["label"]
yTrain = to_categorical(yTrain, num_classes = 25)
yTest = to_categorical(yTest, num_classes = 25)

dataTrain.drop("label",axis=1,inplace=True)
dataTest.drop("label",axis=1,inplace=True)


# In[84]:


## Normalize Data and reshape data
xTrain, xTest = dataTrain.values, dataTest.values
xTrain = xTrain.reshape(-1,28,28,1)
xTest = xTest.reshape(-1,28,28,1)
xTrain = xTrain/255
xTest = xTest/255
xTrain, xDev, yTrain, yDev = train_test_split(xTrain,yTrain,test_size=0.2,random_state=1)


# In[85]:


## Data Augmentation
trainData = ImageDataGenerator(
    rotation_range = 45,
    zoom_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.2,
    horizontal_flip = True
)


# In[86]:


## Model generation
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = "Same", activation = "relu", input_shape = (28,28,1)))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = "Same", activation = "relu"))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(25,activation="softmax"))

model.summary()


# In[87]:


## Model Compile

model.compile(optimizer="adam", loss ="categorical_crossentropy", metrics = ["accuracy"])
epochs = 10
batchSize = 256

history = model.fit_generator(generator = trainData.flow(xTrain,yTrain,batch_size = batchSize), 
                              epochs = epochs, 
                             validation_data=(xDev,yDev))


# In[90]:


# Plots

trainAcc, valAcc = history.history["accuracy"], history.history["val_accuracy"]
trainLoss, valLoss = history.history["loss"], history.history["val_loss"]

plt.plot(range(epochs),trainAcc,"r",label="Training Accuracy")
plt.plot(range(epochs),valAcc,"b",label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()

plt.plot(range(epochs),trainLoss,"r",label="Training Loss")
plt.plot(range(epochs),valLoss,"b",label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()


# In[110]:


## Prediction
preds = model.predict_classes(xTest)
yTest = np.argmax(yTest,1)
confusionMatrix = confusion_matrix(yTest,preds)


# In[125]:


print("Accuracy Of Test Data: {}".format(accuracy_score(yTest,preds)))

classes = ["A","B","C","D","E","F","G","H","I","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y"]
_, ax = plt.subplots(figsize=(15,15))
heatmap = sn.heatmap(confusionMatrix, annot=True, fmt= '.1f',ax=ax)
heatmap.yaxis.set_ticklabels(classes)
heatmap.xaxis.set_ticklabels(classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[ ]:




