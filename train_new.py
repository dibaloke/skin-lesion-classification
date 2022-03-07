import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import datetime
import os
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D
from keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#%%

im_path = 'train'
csv_file_train = pd.read_csv('../ISIC_2019_Training_GroundTruth.csv', delimiter=',')
class_names = list(csv_file_train.columns[1:])

train_datagen = image_dataset_from_directory(im_path, seed=1, batch_size=32, label_mode='categorical', validation_split=.15, 
                                             image_size=(256, 256), subset='training')


val_datagen = image_dataset_from_directory(im_path, seed=1, batch_size=32, label_mode='categorical', validation_split=.15, 
                                             image_size=(256, 256), subset='validation')



#%%

model1 = Sequential()
model1.add(Conv2D(16, kernel_size = (3,3), input_shape = (256, 256, 3), activation = 'relu', padding = 'same'))
model1.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model1.add(MaxPool2D(pool_size = (2,2)))

model1.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model1.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model1.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model1.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model1.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model1.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model1.add(Flatten())

model1.add(Dense(64, activation = 'relu'))
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(9, activation='softmax'))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999,
                                    epsilon = 1e-8)

model1.compile(loss = 'categorical_crossentropy',
             optimizer = optimizer,
              metrics = ['accuracy'])

print(model1.summary())
#%%


model1.fit(train_datagen, epochs=10, callbacks=[learning_rate_reduction])
model1.save('model1.hdf5')

#%%
'''
model2 = Sequential()
model2.add(Conv2D(16, kernel_size = (3,3), input_shape = (576, 768,3), activation = 'relu', padding = 'same'))
model2.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model2.add(MaxPool2D(pool_size = (2,2)))

model2.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model2.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model2.add(Dropout(0.2))
model2.add(MaxPool2D(pool_size = (2,2), padding = 'same'))





model2.add(Flatten())

model2.add(Dense(64, activation = 'relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(3, activation='softmax'))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00075,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999,
                                    epsilon = 1e-8)

model2.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = optimizer,
              metrics = ['accuracy'])

print(model2.summary())

model2 = Sequential()
model2.add(Conv2D(16, kernel_size = (3,3), input_shape = (576, 768, 3), activation = 'relu', padding = 'same'))
model2.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model2.add(MaxPool2D(pool_size = (2,2)))

model2.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model2.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model2.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model2.add(Dropout(0.2))
model2.add(MaxPool2D(pool_size = (2,2), padding = 'same'))





model2.add(Flatten())

model2.add(Dense(64, activation = 'relu'))
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(3, activation='softmax'))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00075,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999,
                                    epsilon = 1e-8)

model2.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = optimizer,
              metrics = ['accuracy'])

print(model2.summary())




model3 = Sequential()
model3.add(Conv2D(16, kernel_size = (3,3), input_shape = (28, 28, 3), activation = 'relu', padding = 'same'))
model3.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model3.add(Dropout(0.2))
model3.add(MaxPool2D(pool_size = (2,2)))

model3.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model3.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model3.add(Dropout(0.2))
model3.add(MaxPool2D(pool_size = (2,2), padding = 'same'))

model3.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model3.add(Conv2D(64, kernel_size = (3,3), activation = 'relu', padding = 'same'))
model3.add(Dropout(0.2))
model3.add(MaxPool2D(pool_size = (2,2), padding = 'same'))





model3.add(Flatten())

model3.add(Dense(64, activation = 'relu'))
model3.add(Dense(64, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(32, activation='relu'))
model3.add(Dense(3, activation='softmax'))


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)


optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00075,
                                    beta_1 = 0.9,
                                    beta_2 = 0.999,
                                    epsilon = 1e-8)

model3.compile(loss = 'sparse_categorical_crossentropy',
             optimizer = optimizer,
              metrics = ['accuracy'])

print(model3.summary())




history = model1.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size = 128,
                    epochs = 30,
                    callbacks=[learning_rate_reduction])


ACC = history.history['accuracy']
VAL_ACC = history.history['val_accuracy']
#model1.save('model1.hdf5')

history2 = model2.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size = 128,
                    epochs = 30,
                    callbacks=[learning_rate_reduction])


ACC = history2.history['accuracy']
VAL_ACC = history2.history['val_accuracy']
#model2.save('model2.hdf5')

history3 = model3.fit(X_train,
                    Y_train,
                    validation_split=0.2,
                    batch_size =128,
                    epochs = 30,
                    callbacks=[learning_rate_reduction])


ACC = history3.history['accuracy']
VAL_ACC = history3.history['val_accuracy']
#model3.save('model3.hdf5')
'''



