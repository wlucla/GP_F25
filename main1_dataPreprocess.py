# this module serves only for Data Preprocessing with APplication of PCA
#It results in 3 relevant files used to train the FFNN model:
#1) DR_test_data.csv (dimensionally reduced training data)
#2) DR_val_data.csv (dimensionally reduced validation data)
#3) DR_test_data.csv (dimensionally reduced test data)
#all other csv files are intermediate and unused.

#NUMBER OF PRINCIPlE COMPONENTS CHOSEN: 20



import data_visualizer
import train_val_test_splitter
import pandas as pd
data = pd.read_csv('digits_dataset.csv')

#visualize a digit arbitratily
data_visualizer.data_visualizer(10, data_path=data)
print('No issues on Visualization')

#split the dataset into train, validation and test sets
train_val_test_splitter.train_val_test_splitter(0.7, 0.15, 0.15, data_path=data)
print('No issues on Splitter')

#test a digit from the train, val, and test set to confirm split worked
train_data = pd.read_csv('train_data.csv')
validation_data = pd.read_csv('validation_data.csv')
test_data = pd.read_csv('test_data.csv')
data_visualizer.data_visualizer(0, data_path=train_data)
data_visualizer.data_visualizer(0, data_path=validation_data)
data_visualizer.data_visualizer(0, data_path=test_data)
print('No issues on Visualization after SPlit')
#no issues on split and visualization logic


#perform PCA dimensionality reduction on train, val, and test sets
#THIS WILL CREATE THE FILES: pca_reduced_data.csv, DR_val_data.csv, DR_test_data.csv
#THESE WILL BE THE FINAL DATA FILES THAT OUR FFNN MODEL WILL TRAIN ON.
from  pca_test_val import pca_test_val
#reducing to 3 latent dimensions
mean, components =pca_test_val(n=20, train=train_data, val=validation_data, test=test_data)
print('No issues PCA Dimensionality Reduction')

#verify that your PCA reduced files actulally look like digits when reconstructed.
# use mean and components on the reduced data to reconstruct a single digit to verify.
reduced_train_data = pd.read_csv('DR_train_data.csv')

from dimensionality_reduced_visualization import dimensionality_reduced_visualization
dimensionality_reduced_visualization(index=1, data_path=reduced_train_data, mean=mean, components=components)
print("No issues with visualization of PCAed data")
#REFER TO README, all unused csv files will be moved to /intermediate_files/ folder
import os
import shutil
intermediate_files = ['train_data.csv', 'validation_data.csv', 'test_data.csv']
os.makedirs('intermediate_files', exist_ok=True)
for file in intermediate_files:
    shutil.move(file, f'intermediate_files/{file}')

print('intermediate files stored')