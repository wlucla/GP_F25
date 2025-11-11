from sklearn.decomposition import PCA
import pandas as pd

def pca_training(latent_dimensions, data):
    
    # Apply PCA to reduce the dimensionality of the dataset.
    # latent_dimensions (int): Number of dimensions to reduce the data to.
    # data_path (DataFrame): The input dataset.

    #you will call this function in another function that dimneisonally reduces the training and validation data. 
    #this is because we need the eigenvectors and mean from the training data


    digits = data.iloc[:,0].values
    features = data.iloc[:,1:].values


    pca_model = PCA(n_components=latent_dimensions)
    latents = pca_model.fit_transform(features)


    features = data.iloc[:, 1:]  # Assuming first column is label
    reduced_features = pca_model.fit_transform(features)

    # Create a new DataFrame with reduced features and original labels
    column_headers = ['latent'+str(1+i) for i in range(latent_dimensions)]
    reduced_data = pd.DataFrame(reduced_features, columns=column_headers) #dataframe with headers
    reduced_data.insert(0, 'label', digits) #insert digits at the start
    
    #write to csv the final preprocessed training data
    reduced_data.to_csv('DR_test.csv', index=False)

    #you should be returning the eigenvectors in order greatest to least. and the mean too
    return pca_model.mean_, pca_model.components_