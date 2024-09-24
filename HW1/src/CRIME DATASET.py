#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


'''
Girish Rajani-Bathija
A20503736
CS 577 - F22
Assignment 1 Part 2 Question 4 - CRIME Dataset
'''

from matplotlib import pyplot
from keras import models, layers
import numpy as np
import matplotlib.pyplot as plt
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split

#Load crime dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"

def load_crime_data(url):
    dataframe = pd.read_csv(url, header=None)
    dataframe = dataframe.replace('?',0)
    X = dataframe.iloc[:, :-1].drop([0, 1, 2, 3, 4], axis=1)
    Y = dataframe.iloc[:, -1]
    train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=0.3)
    return train_data, test_data, train_labels, test_labels

train_data, test_data, train_labels, test_labels = load_crime_data(url)

train_data = train_data.astype('float')
test_data = test_data.astype('float')

#Build the network
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

#Perform k-fold cross validation using 4 folds
k=4
num_val_samples = len(train_data) // k
all_mae_histories = []
num_epochs = 30

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_labels = train_labels[i * num_val_samples: (i + 1) * num_val_samples]
    
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0)
    
    partial_train_labels = np.concatenate(
        [train_labels[:i * num_val_samples],
        train_labels[(i + 1) * num_val_samples:]],
        axis=0)

    
#Train the model
model = build_model()
history = model.fit(partial_train_data, partial_train_labels,
validation_data=(val_data, val_labels),
epochs=num_epochs, batch_size=10, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)

#Record validation mean absolute error
mae_history = history.history['val_mae']
all_mae_histories.append(mae_history)

average_mae_history = [
np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

#Plot results of validation MAE on epochs
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


print('Test MAE Score:',test_mae_score)

#Save weights of model
checkpoint_path = 'crime_model/crime_model_weigths'
if os.path.isfile(checkpoint_path) is False:
    model.save_weights(checkpoint_path)

#Create a new model with same architecture to evaluate using weights of previously trained model
nmodel = build_model()
nmodel.load_weights(checkpoint_path)

new_test_mse_score, new_test_mae_score = nmodel.evaluate(test_data, test_labels)
print('Test MAE Score of new loaded model:',new_test_mae_score)


# In[ ]:




