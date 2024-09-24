#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


'''
Girish Rajani-Bathija
A20503736
CS 577 - F22
Assignment 1 Part 2 Question 3 - SPAM Dataset
'''

from matplotlib import pyplot
from keras.utils.np_utils import to_categorical
from keras import models, layers
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#Load spambase dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"

def load_spam_data(url):
    dataframe = pd.read_csv(url, header=None)
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1]
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.3)
    return train_data, test_data, train_labels, test_labels

train_data, test_data, train_labels, test_labels = load_spam_data(url)

#normalize training and testing data
train_data = preprocessing.normalize(train_data)
test_data = preprocessing.normalize(test_data)

#vectorize labels using categorical encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#Build the network
model = models.Sequential()
model.add(layers.Dense(150, activation='relu', input_shape=(57,)))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(2, activation='sigmoid'))

#Compile the model
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

#Create validation set from training data
x_val = train_data[:500]
partial_x_train = train_data[500:]
y_val = one_hot_train_labels[:500]
partial_y_train = one_hot_train_labels[500:]

#Train the model
history = model.fit(partial_x_train, partial_y_train,
epochs=50, batch_size=100,
validation_data=(x_val, y_val))

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
print('evaluation of final model:' , model.evaluate(test_data, one_hot_test_labels))

#Save weights of model
checkpoint_path = 'spam_model/spam_model_weigths'
if os.path.isfile(checkpoint_path) is False:
    model.save_weights(checkpoint_path)

#Create a new model with same architecture to evaluate using weights of previously trained model
nmodel = models.Sequential()
nmodel.add(layers.Dense(150, activation='relu', input_shape=(57,)))
nmodel.add(layers.Dense(100, activation='relu'))
nmodel.add(layers.Dense(2, activation='sigmoid'))

#Compile the new model
nmodel.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

#Load the weights of the previously trained model and confirm that both models have same accuracy since the same weights are used
nmodel.load_weights(checkpoint_path)
print('evaluation of loaded model:' , nmodel.evaluate(test_data, one_hot_test_labels))


# In[ ]:





# In[ ]:




