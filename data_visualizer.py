import pandas as pd
import matplotlib.pyplot as plt



def data_visualizer(i, data_path='digits_dataset.csv'): 
    #function to visualize the digit at index i (row i) in the dataset (dataset is ordered in no particular order)
    #each row in the dataset corresponds to an unwrapped 28*28 image of drawn digit.
    # data = pd.read_csv('digits_dataset.csv')

    label = data_path.iloc[i, 0] #identity of the digit

    pixels = data_path.iloc[i, 1:].values.reshape(28, 28)

    plt.imshow(pixels, cmap='gray')
    plt.title(f'Label: {label}')
    plt.show()
    return


# data_visualizer(3, data_path=pd.read_csv('digits_dataset.csv'))