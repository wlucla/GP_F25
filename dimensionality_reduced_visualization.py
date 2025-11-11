#takes an input dataset, mean, and components, and index of a digit to reconstruct
import numpy as np
import matplotlib.pyplot as plt

def dimensionality_reduced_visualization(index, data_path, mean, components):

    #digits reduced representation
    reduced_digit = data_path.iloc[index, 1:].values  #irst column==label

    #reconstruct the digit from its reduced representation
    reconstructed_digit = mean + components.T@reduced_digit.T

    reconstructed_image = reconstructed_digit.reshape(28, 28)

    #plot the reconstructed image
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstructed Digit at csv Index {index}')
    plt.axis('off')
    plt.show()