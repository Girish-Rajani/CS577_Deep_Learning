#!/usr/bin/env python
# coding: utf-8

# In[24]:


'''
Girish Rajani-Bathija
A20503736
CS 577 - F22
Assignment 1 Part 2 Question 2 - CIFAR Dataset
'''

from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import os.path

# load dataset
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()


#Using the subset of the first 3 classes with indices 0,1,2

#Creating first subset from the plane class, index 0
index_0 = np.where(train_labels == 0)[0]
train_data_0 = train_data[index_0]
train_label_0 = train_labels[index_0]

index_0 = np.where(test_labels == 0)[0]
test_data_0 = test_data[index_0]
test_label_0 = test_labels[index_0]

#Creating first subset from the automobile class, index 1
index_1 = np.where(train_labels == 1)[0]
train_data_1 = train_data[index_1]
train_label_1 = train_labels[index_1]

index_1 = np.where(test_labels == 1)[0]
test_data_1 = test_data[index_1]
test_label_1 = test_labels[index_1]

#Creating first subset from the bird class, index 2
index_2 = np.where(train_labels == 2)[0]
train_data_2 = train_data[index_2]
train_label_2 = train_labels[index_2]

index_2 = np.where(test_labels == 2)[0]
test_data_2 = test_data[index_2]
test_label_2 = test_labels[index_2]

#Concatenate all subsets to create a set of training and testing data with only those 3 subsets
train_data = np.concatenate((train_data_0, train_data_1, train_data_2), axis=0)
train_labels = np.concatenate((train_label_0, train_label_1, train_label_2), axis=0)

test_data = np.concatenate((test_data_0, test_data_1, test_data_2), axis=0)
test_labels = np.concatenate((test_label_0, test_label_1, test_label_2), axis=0)



#vectorize training and testing data 
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

train_data = train_data/255.0
test_data = test_data/255.0

#vectorize labels using categorical encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


#Build the network
model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

#Compile the model
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#Create validation set from training data
train_val = train_data[:3000]
partial_train_data = train_data[3000:]
label_val = train_labels[:3000]
partial_train_labels = train_labels[3000:]

#Train the model
history = model.fit(partial_train_data, partial_train_labels,
epochs=50, batch_size=256,
validation_data=(train_val, label_val))

    
#Plot training and validation loss graph as a function of epochs
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Plot training and validation accuracy
plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Evaluate accuracy of final model
print('evaluation of final model:' , model.evaluate(test_data, test_labels))

#Save weights of model
checkpoint_path = 'cifar_model/cifar_model_weigths'
if os.path.isfile(checkpoint_path) is False:
    model.save_weights(checkpoint_path)

#Create a new model to evaluate using weights of previously trained model
nmodel = models.Sequential()
nmodel.add(layers.Flatten(input_shape=(32, 32, 3)))
nmodel.add(layers.Dense(64, activation='relu'))
nmodel.add(layers.Dense(32, activation='relu'))
nmodel.add(layers.Dense(3, activation='softmax'))

#Compile the new model
nmodel.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

#Load the weights of the previously trained model and confirm that both models have same accuracy since the same weights are used
nmodel.load_weights(checkpoint_path)
print('evaluation of loaded model:' , nmodel.evaluate(test_data, test_labels))


# In[ ]:





# In[ ]:




