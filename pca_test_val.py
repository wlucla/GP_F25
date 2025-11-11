#this will generate the complete dimensionally reduced datasets for trian, test, and val
from sklearn.decomposition import PCA
import pandas as pd
import pca_training

#params to take: latent_dimension=n, data_path for train val, test
def pca_test_val(n, train, val, test):
    #get the mean and eigenvectors from training data

    mean, components = pca_training.pca_training(n, train)

    #reduce dimensionality of validation data
    val_digits = val.iloc[:,0].values
    val_features = val.iloc[:,1:].values

    #transform validation data using the PCA model from training data
    latent_val_features = (val_features - mean)@(components.T)
    column_headers = ['latent'+str(1+i) for i in range(n)]
    reduced_val_data = pd.DataFrame(latent_val_features, columns=column_headers)
    reduced_val_data.insert(0, 'label', val_digits)
    reduced_val_data.to_csv('DR_val_data.csv', index=False)




    #reduce dimensionality of test data
    test_digits = test.iloc[:,0].values
    test_features = test.iloc[:,1:].values
    latent_test_features = (test_features - mean)@(components.T)
    column_headers1 = ['latent'+str(1+i) for i in range(n)]
    reduced_test_data = pd.DataFrame(latent_test_features, columns=column_headers1)
    reduced_test_data.insert(0, 'label', test_digits)
    reduced_test_data.to_csv('DR_test_data.csv', index=False)

    return mean, components

