import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_test_splitter(train_percent, val_percent, test_percent, data_path):
    
    #Splits the dataset into training, validation, and test sets based on the provided percentages.
    #train_percent (float): Percentage of data to be used for training.
    #val_percent (float): Percentage of data to be used for validation.
    #test_percent (float): Percentage of data to be used for testing.

    #split out training data
    train_percent = float(train_percent)
    val_percent = float(val_percent)
    test_percent = float(test_percent)

    train_data, dummy_data = train_test_split(data_path, test_size=test_percent+val_percent, random_state=42, shuffle=True)

    validation_data, test_data = train_test_split(dummy_data, test_size=test_percent/(test_percent+val_percent), random_state=42, shuffle=True)

    #write to 3 csv files
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    validation_data.to_csv('validation_data.csv', index=False)

    print("Files Created")


    return