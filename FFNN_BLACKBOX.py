#What you need to know.
#### Load_Data(data_path): loads a csv file and returns features and labels as numpy arrays. YOU SHOULD USE DR_train_data.csv, DR_val_data.csv as data_path inputs.
#### create_dataloader(train_features, train_labels, val_features, val_labels, test_features, test_labels, batch_size): creates pytorch dataloaders for train, val, test datasets out of the Load_Data outputs
#### model, gradient_descent_optimizer, criterion =initialize_network(#input_features, #hidden_neurons, #n_layers, #output_neurons, activation_function, learningRate): initializes the n-layered feedforward neural network with the specified hyperparameters and returns the model, optimizer, and loss criterion.
#### training(train_loader, val_loader, model, optimizer, criterion, epochs): Trains the model using the provided dataloaders, model, optimizer, and criterion for the specified number of epochs. Outputs accuracy plots and loss plots for training and validation sets.

#logistic regression with softmax output layer 0-9
from sklearn.model_selection import train_test_split
import pandas
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

def Load_Data(data_path):
    data = pandas.read_csv(data_path)
    labels = data.iloc[:, 0].values.astype(np.int64)
    features = data.iloc[:, 1:].values.astype(np.float32)
    #features should be Nxn where N is number of samples, n is number of features
    return features, np.array([[_] for _ in labels])


#create dataloaders for easier iteration during sgd later
def create_dataloader(train_features, train_labels, val_features, val_labels, test_features, test_labels, batch_size):
    train_dataset = TensorDataset(torch.tensor(train_features), torch.tensor(train_labels).squeeze())
    val_dataset = TensorDataset(torch.tensor(val_features), torch.tensor(val_labels).squeeze())
    test_dataset = TensorDataset(torch.tensor(test_features), torch.tensor(test_labels).squeeze())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


#make a pretty black-boxed loss evaluator to be used on the val and test sets
def loss_evaluation(model, data_loader, criterion):
    model.eval()
    loss = 0.00
    for feature, labels in data_loader:
        outputs = model(feature)
        loss_per_batch = criterion(outputs, labels)
        loss += loss_per_batch.item()
    return loss/len(data_loader)

def accuracy_evaluation(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    for features, labels in data_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1) #the preduction is the class with highest score. Keep the class index
        total += labels.shape[0] #add bathc size to total
        correct += (predicted == labels).sum().item() #record number of correct predictions.
    return correct / total


#n-layered nueral network with ??? activation funcntions for the hidden layers
class FFNN_n_layers(torch.nn.Module):
    def __init__(self, input_features, hidden_neurons, n_layers, output_neurons,activation_function):
        def activator(activation_function):
            if activation_function == 'relu':
                return torch.nn.ReLU()
            elif activation_function == 'tanh':
                return torch.nn.Tanh()
            elif activation_function == 'sigmoid':
                return torch.nn.Sigmoid()
            else:
                raise ValueError("idk that activation function")
        super(FFNN_n_layers, self).__init__()
        self.layers = torch.nn.ModuleList()

        #define input layer
        self.layers.append(torch.nn.Linear(input_features, hidden_neurons))
        self.layers.append(activator(activation_function))
        #define hidden layers, 2 less than n_layers because input and output layers are separately implmeented
        for _ in range(n_layers - 2):
            self.layers.append(torch.nn.Linear(hidden_neurons, hidden_neurons))
            self.layers.append(activator(activation_function))
        
        #define output layer
        self.layers.append(torch.nn.Linear(hidden_neurons, output_neurons))
       
            
    def forward(self, x):
        #pass though all layers except output layer
        for layer in self.layers[:-1]:
            x = layer(x)
        #pass to output
        x = self.layers[-1](x)
        #SOFTMAX NOT HERE, WILL BE IN LOSS FUNCTION by crossentropy loss
        return x
        
def initialize_network(input_features, hidden_neurons, n_layers, output_neurons, activation_function, LR):
        model = FFNN_n_layers(input_features, hidden_neurons, n_layers, output_neurons, activation_function)
        gradient_descent_optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()
        return model, gradient_descent_optimizer, criterion
    

def training(train_loader, val_loader, model, optimizer, criterion, epochs):

        training_loss=[]
        validation_loss=[]
        training_accuracy=[]
        validation_accuracy=[]
        for epoch in range(epochs):
            model.train()
            for features, labels in train_loader:
                optimizer.zero_grad() #delete remaining stuff
                predict = model(features)
                loss = criterion(predict, labels)
                loss.backward()
                optimizer.step()
            
            #evaluate hyperparametenrs on validation set
            val_loss = loss_evaluation(model, val_loader, criterion)
            train_loss= loss_evaluation(model, train_loader, criterion)
            training_loss.append(train_loss)
            validation_loss.append(val_loss)



            training_acc = accuracy_evaluation(model, train_loader)
            validation_acc = accuracy_evaluation(model, val_loader)
            training_accuracy.append(training_acc)
            validation_accuracy.append(validation_acc)
        plt.figure(1)
        # plt.plot(range(epochs), training_loss, label='Training Loss')
        # plt.plot(range(epochs), validation_loss, label='Validation Loss')
        # plt.title('Loss vs Epochs')
        # plt.legend()
        # plt.show()
        # plt.figure(2)
        # plt.plot(range(epochs), training_accuracy, label='Training Accuracy')
        # plt.plot(range(epochs), validation_accuracy, label='Validation Accuracy')
        # plt.title('Accuracy vs Epochs')
        # plt.legend()
        # plt.show()
        
        return training_accuracy, validation_accuracy, training_loss, validation_loss

def test_set_eval(train_loader, test_loader, model, optimizer, criterion, epochs):

        training_loss=[]
        test_loss=[]
        training_accuracy=[]
        test_accuracy=[]
        for epoch in range(epochs):
            model.eval()
            for features, labels in train_loader:
                optimizer.zero_grad() #delete remaining stuff
                predict = model(features)
                loss = criterion(predict, labels)
                loss.backward()
                optimizer.step()
            
            #evaluate hyperparametenrs on validation set
            val_loss = loss_evaluation(model, test_loader, criterion)
            train_loss= loss_evaluation(model, train_loader, criterion)
            training_loss.append(train_loss)
            test_loss.append(val_loss)



            training_acc = accuracy_evaluation(model, train_loader)
            validation_acc = accuracy_evaluation(model, test_loader)
            training_accuracy.append(training_acc)
            test_accuracy.append(validation_acc)
        # plt.figure(1)
        # plt.plot(range(epochs), training_loss, label='Training Loss')
        # plt.plot(range(epochs), test_loss, label='Test Loss')
        # plt.title('Loss vs Epochs')
        # plt.legend()
        # plt.show()
        plt.figure(1)
        plt.plot(range(epochs), training_accuracy, label='Training Accuracy', color='purple')
        plt.plot(range(epochs), test_accuracy, label='Test Accuracy', color='red')
        plt.title('Results of Best Hyperparameters on Test Data Set')
        plt.legend()
        plt.show()
        
        print(f"Final training accuracy = {training_accuracy[-1]}\n")
        print(f"Final test accuracy = {test_accuracy[-1]}")
              
        return training_accuracy[-1], test_accuracy[-1]




# # #example usage for random hyperparameters...
# # np.random.seed(42)
# # torch.manual_seed(42)

# # batch_size = 512

# # #load data
# train_features, train_labels = Load_Data('DR_train_data.csv')
# val_features, val_labels = Load_Data('DR_val_data.csv')
# test_features, test_labels = Load_Data('DR_test_data.csv')

# train_loader, val_loader, test_loader = create_dataloader(train_features, train_labels, val_features, val_labels, test_features, test_labels, batch_size)

# # #try some hyperparameters to make sure it even works
# LR = 0.005
# input_features = train_features.shape[1]
# hidden_neurons = 25
# n_layers = 4
# output_neurons = 10
# activation_function = 'tanh'
# epochs = 50
# model, optimizer, criterion = initialize_network(input_features, hidden_neurons, n_layers, output_neurons, activation_function, LR)

# results = training(train_loader, val_loader, model, optimizer, criterion, epochs)
