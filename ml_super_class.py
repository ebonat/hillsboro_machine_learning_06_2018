
import os
import sys
import traceback
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier    
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class MLSuperClass(object):
    '''
    machine learning super class
    '''
    def __init__(self):
        '''
        super class constructor
        '''
    
    def read_data_file(self, data_file_name, column_name=None, encoding_unicode=None):
        '''
        read data from a csv file
        :param file_name: csv file path and name
        :param column_name: csv file columns name
        :param encoding_unicode: csv file encoding unicode
        :return data frame
        '''      
        try:
            project_directory_path = os.path.dirname(sys.argv[0])  
            csv_file_path_name = os.path.join(project_directory_path, data_file_name)  
            if column_name is None:
                df_file_name = pd.read_csv(filepath_or_buffer=csv_file_path_name, sep=",", encoding=encoding_unicode)   
            else:
                df_file_name = pd.read_csv(filepath_or_buffer=csv_file_path_name, sep=",", names=column_name, encoding=encoding_unicode)               
        except Exception:
            self.print_exception_message()
        return df_file_name
    
    def read_image_file(self, data_folder_path, data_file_name):        
        try:
            csv_file_path_name = os.path.join(data_folder_path, data_file_name)
            df_file_name = pd.read_csv(filepath_or_buffer=csv_file_path_name, header=None)
        except Exception:
            self.print_exception_message()
        return df_file_name
    
    def print_status(self, print_message):
        try:
            if print_message is not None:
                print(print_message)
                print()
        except Exception:
            self.print_exception_message()
            
    def show_file_information(self, df_file_name):
        '''
        show data file information
        :param df_file_name: data frame
        :return none
        '''
        try:         
            if df_file_name is not None:
                df_file_name.info()
                print()
        except Exception:
            self.print_exception_message()
            
    def show_file_data(self, df_file_name):
        '''
        print data file
        :param df_file_name: data frame
        :return none
        '''
        try:
            if df_file_name is not None:
                print(df_file_name)
            print()
        except Exception:
            self.print_exception_message()

    def select_y_x_image(self, df_file_name, target_column_number):
        '''
        select x label and y target data frames by target column
        :param df_file_name: data frame
        :param target_column_number: target column name
        :return y and x data frames
        '''
        try:
            df_file_name_columns = df_file_name.shape[1]          
            y = df_file_name.iloc[:,0:target_column_number]            
            X = df_file_name.iloc[:,target_column_number:df_file_name_columns]    
        except Exception:
            self.print_exception_message()
        return y, X
  
    def select_y_x_data(self, df_file_name, target_column_name):
        '''
        select x label and y target data frames by target column
        :param df_file_name: data frame
        :param target_column_name: target column name
        :return y and x data frames
        '''
        try:
            y = df_file_name[target_column_name]     
            X = df_file_name.drop(labels=target_column_name, axis=1)                   
        except Exception:
            self.print_exception_message()
        return y, X
    
    def train_test_split_data(self, X, y, test_size_percentage, random_state, stratify=None):       
        '''
        select x and y train and test data
        :param X: x label data frame
        :param Y: y label data frame
        :param test_size_percentage: test size in percentage (%)
        :param random_state: random state initial value
        :param stratify: y stratify data
        return: x and y train and test data
        '''
        try:
            if stratify is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage/100, stratify=y, random_state=random_state)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percentage/100, random_state=random_state)                
        except Exception:
            self.print_exception_message()
        return X_train, X_test, y_train, y_test

    def train_valid_test_split_data(self, X, y, test_size, test_size_percentage, stratify=None):
#         20% 80/10/10 (test/valid/test)
#         30% 70/15/15
#         40% 60/20/20
#         50% 50/25/25
        try:
            if stratify is not None:
                X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=test_size_percentage/100, stratify=y, random_state=1)        
                X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=test_size_percentage/100, stratify=y_test_valid, random_state=1)        
            else:
                X_train, X_test_valid, y_train, y_test_valid = train_test_split(X, y, test_size=test_size_percentage/100, random_state=1)        
                X_valid, X_test, y_valid, y_test = train_test_split(X_test_valid, y_test_valid, test_size=test_size_percentage/100, random_state=1)      
        except:
            self.print_exception_message()
        return X_train, y_train, X_valid, y_valid, X_test, y_test
     
    def convert_data_type(self, df_data, data_type):  
        try:         
            df_data = df_data.astype(data_type)
        except:
            self.print_exception_message()
        return df_data

    def data_min_max_scale(self, df_data, x_min, x_max):
        try:       
            df_data = (df_data - x_min) / (x_max - x_min)
        except:
            self.print_exception_message()
        return df_data           
    
    def data_from_2d_to_4d_array(self, df_data, image_width, image_height):
        try:   
            df_data = df_data.values.reshape(df_data.shape[0], image_width, image_height, 1)
        except:
            self.print_exception_message()
        return df_data      
    
    def standard_scaler_data(self, X_label_train, X_label_test):
        '''
        select x label train and test scaled
        :param X_label_train: x label train data frame
        :param X_label_test: x label test data frame
        :return x label train and test scaled
        '''
        try:
            scaler = StandardScaler()
            scaler.fit(X_label_train)
            X_label_train_scaled = scaler.transform(X_label_train)
            X_label_test_scaled = scaler.transform(X_label_test)
        except Exception:
            self.print_exception_message()
        return X_label_train_scaled, X_label_test_scaled
 
    def training_model(self, X_label_train_scaled, Y_label_train, hidden_layer_neuron_sizes, activation_function, solver_optimization, maximum_iteration, random_state):        
        '''
        create and fit the multi-layer perceptron classifier (model)
        :param X_label_train_scaled: x label train scaled
        :param Y_label_train: y label train
        :param hidden_layer_neuron_sizes: hidden layer neuron sizes
        :param activation_function: activation function
        :param solver_optimization: solver optimization
        :param maximum_iteration: maximum iteration
        :param random_state: random state initial value
        :return multi-layer perceptron classifier (model)
        '''
        try:
            mlp_classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_neuron_sizes, activation=activation_function, solver=solver_optimization, max_iter=maximum_iteration, random_state=random_state)
            mlp_classifier.fit(X_label_train_scaled, Y_label_train)
        except Exception:
            self.print_exception_message()
        return mlp_classifier    
    
    def predicting_model(self, mlp_classifier, X_label_test_scaled):
        '''
        select y target predicted data frame
        :param mlp_classifier: multi-layer perceptron classifier 
        :param X_label_test_scaled: x label test scaled
        :return y target predicted
        '''
        try:
            Y_target_predicted = mlp_classifier.predict(X_label_test_scaled)
        except Exception:
            self.print_exception_message()
        return Y_target_predicted
    
    def evaluating_model(self, Y_target_test, Y_target_predicted):
        '''
        print confusion matrix, classification report and accuracy score values
        :param Y_target_test: y target test
        :param Y_target_predicted: y target predicted
        :return none
        '''
        try:
            accuracy_score_value = accuracy_score(Y_target_test, Y_target_predicted) * 100
            accuracy_score_value = float("{0:.2f}".format(accuracy_score_value))
            print("Accuracy Score:")        
            print( "{} %".format(accuracy_score_value))
            print()
            
            confusion_matrix_value = confusion_matrix(Y_target_test, Y_target_predicted)
            print("Confusion Matrix:")
            print(confusion_matrix_value)
            print()
            
            classification_report_result = classification_report(Y_target_test, Y_target_predicted)        
            print('Classification Report:')
            print(classification_report_result)
            print()                    
        except Exception:
            self.print_exception_message()
            
    def print_exception_message(self, message_orientation="horizontal"):
        """
        print full exception message
        :param message_orientation: horizontal or vertical
        :return none
        """
        try:
            exc_type, exc_value, exc_tb = sys.exc_info()            
            file_name, line_number, procedure_name, line_code = traceback.extract_tb(exc_tb)[-1]       
            time_stamp = " [Time Stamp]: " + str(time.strftime("%Y-%m-%d %I:%M:%S %p")) 
            file_name = " [File Name]: " + str(file_name)
            procedure_name = " [Procedure Name]: " + str(procedure_name)
            error_message = " [Error Message]: " + str(exc_value)        
            error_type = " [Error Type]: " + str(exc_type)                    
            line_number = " [Line Number]: " + str(line_number)                
            line_code = " [Line Code]: " + str(line_code) 
            if (message_orientation == "horizontal"):
                print( "An error occurred:{};{};{};{};{};{};{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
            elif (message_orientation == "vertical"):
                print( "An error occurred:\n{}\n{}\n{}\n{}\n{}\n{}\n{}".format(time_stamp, file_name, procedure_name, error_message, error_type, line_number, line_code))
            else:
                pass                    
        except:
            exception_message = sys.exc_info()[0]
            print("An error occurred. {}".format(exception_message))           