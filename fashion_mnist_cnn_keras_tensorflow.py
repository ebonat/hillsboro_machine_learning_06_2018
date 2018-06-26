
# Convolutional Neural Networks in Python with Keras
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python

import time
import numpy as np
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
#     print("NumPy version: {}".format(np.__version__))
#     print("scikit-learn version: {}".format(sklearn.__version__))
#     print("Keras version: {}".format(keras.__version__))
# #     print("TensoFflow version: {}".format(tensorflow.__version__))
#     print("TensoFflow version: {}".format(tensorflow.VERSION))
#     exit()
    
#     1. load the data
    print("load the data...")
    (train_X,train_Y), (test_X,test_Y) = fashion_mnist.load_data()     
    print('Training data shape : ', train_X.shape, train_Y.shape)
    print('Testing data shape : ', test_X.shape, test_Y.shape)
    
#     2. analyze the data
    print("analyze the data...")
    classes = np.unique(train_Y)
    nClasses = len(classes)
    print('Total number of Y classes : ', nClasses)
    print('Y classes : ', classes)
    
#     3. data visualization
    print("visualize the data")
#     display the first image in trainig data
    plt.figure(figsize=[5,5])
    plt.subplot(121)
    plt.imshow(train_X[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(train_Y[0]))    
#     display the first image in testing data
    plt.subplot(122)
    plt.imshow(test_X[0,:,:], cmap='gray')
    plt.title("Ground Truth : {}".format(test_Y[0]))
    plt.show()

#     4. data preprocessing
    print("data preprocessing...")
    train_X = train_X.reshape(-1, 28,28, 1)
    test_X = test_X.reshape(-1, 28,28, 1)
    print(train_X.shape, test_X.shape)
    
#     convert to float32 and apply min-max scale
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = (train_X - 0.0) / (255.0 - 0.0)
    test_X = (test_X - 0.0) / (255.0 - 0.0)
    
#     change the labels from categorical to one-hot encoding
    train_Y_one_hot = to_categorical(train_Y)
    test_Y_one_hot = to_categorical(test_Y)    
#     display the change for category label using one-hot encoding vector
    print('Original Y label:', train_Y[0])
    print('After conversion to Y one-hot encoding vector:', train_Y_one_hot[0])
    
#     data split train/valid
    print("data split train/valid...")
    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=1)
    print(train_X.shape,valid_X.shape,train_Y.shape,valid_Y.shape)
    
#     model the data
    print("model the data...")
    batch_size = 64
    epochs = 20
    num_classes = 10
    
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(28,28,1),padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(Dense(num_classes, activation='softmax'))    
    print(fashion_model.summary())
    
#     compile the model
    print("compile the model...")
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
#     train the model
    print("train the model...")
    fashion_train = fashion_model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(valid_X, valid_Y))
    
#     test model evaluation
    print("test model evaluation...")
    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    
#     plot the accuracy and loss plots between training and validation data
    print("plot the accuracy and loss plots between training and validation data...")
    accuracy = fashion_train.history['acc']
    val_accuracy = fashion_train.history['val_acc']
    loss = fashion_train.history['loss']
    val_loss = fashion_train.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
#     adding dropout into the network
    batch_size = 64
    epochs = 20
    num_classes = 10
    
    fashion_model = Sequential()
    fashion_model.add(Conv2D(32, kernel_size=(3, 3),activation='linear',padding='same',input_shape=(28,28,1)))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D((2, 2),padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Dropout(0.25))
    fashion_model.add(Conv2D(128, (3, 3), activation='linear',padding='same'))
    fashion_model.add(LeakyReLU(alpha=0.1))                  
    fashion_model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
    fashion_model.add(Dropout(0.4))
    fashion_model.add(Flatten())
    fashion_model.add(Dense(128, activation='linear'))
    fashion_model.add(LeakyReLU(alpha=0.1))           
    fashion_model.add(Dropout(0.3))
    fashion_model.add(Dense(num_classes, activation='softmax'))
    print(fashion_model.summary())
    
    fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    
    fashion_train_dropout = fashion_model.fit(train_X, train_Y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(valid_X, valid_Y))
    
    fashion_train_dropout.save("fashion_model_dropout.h5py")
    
    test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=1)
    print('Test loss:', test_eval[0])
    print('Test accuracy:', test_eval[1])
    
    accuracy = fashion_train_dropout.history['acc']
    val_accuracy = fashion_train_dropout.history['val_acc']
    loss = fashion_train_dropout.history['loss']
    val_loss = fashion_train_dropout.history['val_loss']
    epochs = range(len(accuracy))
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
#     predict labels = confution matrix -------------------------------------------------------------------------------
    predicted_classes = fashion_model.predict(test_X)    
    predicted_classes = np.argmax(np.round(predicted_classes),axis=1)    
    print(predicted_classes.shape, test_Y.shape)
    
    correct = np.where(predicted_classes==test_Y)[0]
    print("Found %d correct labels" % len(correct))
    for i, correct in enumerate(correct[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[correct].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_Y[correct]))
        plt.tight_layout()
    
    incorrect = np.where(predicted_classes!=test_Y)[0]
    print("Found %d incorrect labels" % len(incorrect))
    for i, incorrect in enumerate(incorrect[:9]):
        plt.subplot(3,3,i+1)
        plt.imshow(test_X[incorrect].reshape(28,28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_Y[incorrect]))
        plt.tight_layout()
#     -----------------------------------------------------------------------------------------------------------------------------

#     classification report    
    target_names = ["Class {}".format(i) for i in range(num_classes)]
    classification_report_values = classification_report(test_Y, predicted_classes, target_names=target_names)
    print(classification_report_values)
    print()
    
#     confusion matrix
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
    