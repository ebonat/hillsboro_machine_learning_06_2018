
# Convolutional Neural Networks in Python with Keras
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow
import tensorflow as tf

import keras
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

def main():
#     libraries version
#     print("NumPy version: {}".format(np.__version__))
#     print("Pandas version: {}".format(pd.__version__))
#     print("scikit-learn version: {}".format(sklearn.__version__))
#     print("Keras version: {}".format(keras.__version__))
# #     print("TensoFflow version: {}".format(tensorflow.__version__))
#     print("TensoFflow version: {}".format(tensorflow.VERSION))
#     exit()
    
#     data_folder_path = r"E:\Visual WWW\Python\05 DATA PROJECTS\Fashion MNIST images"  #Thermo Fisher
    data_folder_path = r"F:\Visual WWW\Python\05 DATA PROJECTS\Fashion MNIST images"  #Home
    
    print("loading the train data...")
    fashion_mnist_train = "fashion_mnist_train.csv"
    df_fashion_mnist_train = pd.read_csv(os.path.join(data_folder_path, fashion_mnist_train), header=None)
    
    print("loading the test data...")
    fashion_mnist_test = "fashion_mnist_test.csv"
    df_fashion_mnist_test = pd.read_csv(os.path.join(data_folder_path, fashion_mnist_test), header=None)
    
#     train and test columns number
    df_fashion_mnist_train_columns = df_fashion_mnist_train.shape[1]
    df_fashion_mnist_test_columns = df_fashion_mnist_test.shape[1]
        
    target_column_number = 1    
    train_Y = df_fashion_mnist_train.iloc[:,0:target_column_number]
    train_X = df_fashion_mnist_train.iloc[:,target_column_number:df_fashion_mnist_train_columns]
    print("training data shape: {} {}".format(train_X.shape, train_Y.shape))
    
    test_Y = df_fashion_mnist_test.iloc[:,0:target_column_number]
    test_X = df_fashion_mnist_test.iloc[:,target_column_number:df_fashion_mnist_test_columns]    
#     print('Testing data shape: ', test_X.shape, test_Y.shape)   
    print("testing data shape: {} {}".format(test_X.shape, test_Y.shape))
    
    train_Y_classes = np.unique(train_Y)
    print("total number of Y classes: {}".format(train_Y_classes))
       
#     print("visualize the data")
# #     display the first image in trainig data
#     plt.figure(figsize=[5,5])
#     plt.subplot(121)
#     plt.imshow(train_X[0,:,:], cmap='gray')
#     plt.title("Ground Truth : {}".format(train_Y[0]))    
# #     display the first image in testing data
#     plt.subplot(122)
#     plt.imshow(test_X[0,:,:], cmap='gray')
#     plt.title("Ground Truth : {}".format(test_Y[0]))
#     plt.show()

#     convert x train and test from 2d  to 4d array
    train_X = train_X.values.reshape(train_X.shape[0], 28, 28, 1)
    test_X = test_X.values.reshape(test_X.shape[0], 28, 28, 1)
    
#     convert to float32 and apply min-max scale
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = (train_X - 0.0) / (255.0 - 0.0)
    test_X = (test_X - 0.0) / (255.0 - 0.0)
    
#     change the labels from categorical data to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)    
#     display the change for category label using one-hot encoding vector
    print('original Y label:', train_Y[0])
    print('after conversion to Y one-hot encoding vector:', train_Y_one_hot[0])
    
#     data split train/valid
    print("data splitting in train/valid...")
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y_one_hot, test_size=0.4, random_state=1)
    print(train_X.shape, valid_X.shape, train_Y.shape, valid_Y.shape)    
    print("train split data shape: {} {}".format(train_X.shape, train_Y.shape))
    print("valid split data shape: {} {}".format(valid_X.shape, valid_Y.shape))
    
#     adding dropout into the network
    batch_size = 64
    epochs = 20
    num_classes = 10
    
    print("building the model...")
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(28,28,1)))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))           
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    print(fashion_model.summary())
    
    print("compiling the model...")
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    print("fitting the model...")
    fashion_train_dropout = fashion_model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(valid_X, valid_Y))
    
    print("saving the model...")
    fashion_model.save("fashion_model_dropout.h5py")
    
    print("evaluating the model...")
    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
    print('test loss:', test_eval[0])
    print('test accuracy:', test_eval[1])
    
#     print("plotting training/validation accuracy and training/validation lost...")
#     accuracy = fashion_train_dropout.history['acc']
#     val_accuracy = fashion_train_dropout.history['val_acc']
#     loss = fashion_train_dropout.history['loss']
#     val_loss = fashion_train_dropout.history['val_loss']
#     epochs = range(len(accuracy))    
#     plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
#     plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
#     plt.title('Training and validation accuracy')
#     plt.legend()
#     plt.figure()
#     plt.plot(epochs, loss, 'bo', label='Training loss')
#     plt.plot(epochs, val_loss, 'b', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.legend()
#     plt.show()
    
#     predict labels = confution matrix -------------------------------------------------------------------------------
    predicted_classes = fashion_model.predict(test_X)    
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)    
    print(predicted_classes.shape, test_Y.shape)
    
#     correct = np.where(predicted_classes==test_Y)[0]
#     print("Found %d correct labels" % len(correct))
#     for i, correct in enumerate(correct[:9]):
#         plt.subplot(3,3,i+1)
#         plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
#         plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
#         plt.tight_layout()
#     
#     incorrect = np.where(predicted_classes!=test_Y)[0]
#     print("Found %d incorrect labels" % len(incorrect))
#     for i, incorrect in enumerate(incorrect[:9]):
#         plt.subplot(3,3,i+1)
#         plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
#         plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
#         plt.tight_layout()
#     -----------------------------------------------------------------------------------------------------------------------------

#     classification report    
    print("priting classification report...")
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    classification_report_values = classification_report(test_Y, predicted_classes, target_names=target_names)
    print(classification_report_values)
    print()
    
#     confusion matrix
    print("priting confusion matrix...")
    confusion_matrix_result = confusion_matrix(test_Y, predicted_classes)    
    print(confusion_matrix_result)
    print()
    
    
    
    print("Done!")
        
if __name__ == '__main__':    
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print()
    print("Program Runtime:")
    print("Seconds: {} | Minutes: {}".format(seconds, minutes))
    