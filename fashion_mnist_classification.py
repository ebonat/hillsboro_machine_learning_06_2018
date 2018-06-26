
import time
from ml_base_class import MLBaseClass
import config

def main():
    ml_base_class = MLBaseClass()
    
    ml_base_class.print_status("reading fashion_mnist_train.csv file...")
    df_fashion_mnist_train = ml_base_class.read_image_file(config.DATA_FOLDER_PATH, config.FASHION_MNIST_TRAIN)
    
    ml_base_class.print_status("reading fashion_mnist_test.csv file...")
    df_fashion_mnist_test = ml_base_class.read_image_file(config.DATA_FOLDER_PATH, config.FASHION_MNIST_TEST)
    
    ml_base_class.print_status("selecting train y and x...")
    y_train, X_train = ml_base_class.select_y_x_image(df_fashion_mnist_train, config.TARGET_COLUMN_NUMBER)
    
    ml_base_class.print_status("selecting test y and x...")
    y_test, X_test = ml_base_class.select_y_x_image(df_fashion_mnist_test, config.TARGET_COLUMN_NUMBER)
    
    ml_base_class.print_status("converting from 2d to 4d x train and test...")   
    X_train = ml_base_class.data_from_2d_to_4d_array(X_train, config.FASHION_MNIST_IMAGE_WDTH, config.FASHION_MNIST_IMAGE_HEIGHT)
    X_test = ml_base_class.data_from_2d_to_4d_array(X_test, config.FASHION_MNIST_IMAGE_WDTH, config.FASHION_MNIST_IMAGE_HEIGHT)
    
#     ml_base_class.print_status("data splitting in train and valid...")       
#     test_size = 20
#     X_train, X_valid, y_train, y_valid = ml_base_class.train_test_split_data(X, y, test_size, config.RANDOM_STATE)
#      
#     ml_base_class.print_status("data convertion x train and x valid...")      
#     X_train = ml_base_class.convert_data_type(X_train, config.DATA_FLOAT_32)
#     X_valid = ml_base_class.convert_data_type(X_valid, config.DATA_FLOAT_32)
#  
#     ml_base_class.print_status("data min/max scaling...")      
#     X_train = ml_base_class.data_min_max_scale(X_train, config.XMIN, config.XMAX)
#     print(X_train)
    


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    seconds = str(round(end_time - start_time, 1))
    minutes = str(round((end_time - start_time) / 60, 1))
    print()
    print("Program Runtime:")
    print("Seconds: {} | Minutes: {}".format(seconds, minutes))