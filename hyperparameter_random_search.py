import FFNN_BLACKBOX as ffnn
import numpy as np
import random
import matplotlib.pyplot as plt


def hyperparameter_random_search(train_features,train_labels,val_features, val_labels,test_features, test_labels, trials):
    top_val_accuracy = []
    top_train_accuracy = []
    top_accuracy = 0.0
    best_hyperparameters = {
        'learning_rate': None,
        'hidden_neurons': None,
        'n_layers': None,
        'activation_function': None,
        'batch_size': None
    }
    for _ in range(trials):
        #randomly search for hyperparameters using uniform pdf
        lr = 10**random.uniform(-3, -1)  # Learning rate between 0.0001 and 0.1
        hidden_neurons = random.randint(8, 257)  # Hidden neurons count
        n_layers = random.randint(1, 5)  # Number of hidden layers
        activation_function = random.choice(['relu', 'sigmoid', 'tanh']) 
        batch_size = random.randint(8, 200)  # Batch size
        epochs = random.randint(10, 201)  # Number of epochs

        train_loader, val_loader, _ = ffnn.create_dataloader(train_features, train_labels, val_features, val_labels, test_features, test_labels, batch_size)



        model, optimizer, criterion = ffnn.initialize_network(train_loader.dataset.tensors[0].shape[1], hidden_neurons, n_layers, 10, activation_function, lr)
        train_accuracy, val_accuracy,_,_ = ffnn.training(train_loader, val_loader, model, optimizer, criterion, epochs=epochs)  # Get final validation accuracy
        
        if val_accuracy[-1] > top_accuracy:
            top_val_accuracy = val_accuracy
            top_train_accuracy = train_accuracy
            top_accuracy = val_accuracy[-1]
            best_hyperparameters['learning_rate'] = lr
            best_hyperparameters['hidden_neurons'] = hidden_neurons
            best_hyperparameters['n_layers'] = n_layers
            best_hyperparameters['activation_function'] = activation_function
            best_hyperparameters['batch_size'] = batch_size
    
    return best_hyperparameters, top_accuracy, top_train_accuracy, top_val_accuracy

# Example usage:
train_features, train_labels = ffnn.Load_Data('DR_train_data.csv')
val_features, val_labels = ffnn.Load_Data('DR_val_data.csv')
test_features, test_labels = ffnn.Load_Data('DR_test_data.csv')
best_params, best_acc, top_train_accuracy_list, top_val_accuracy_list = hyperparameter_random_search(train_features, train_labels,val_features, val_labels,test_features, test_labels, trials=100)
print("Best Hyperparameters:", best_params)
print("Best Validation Accuracy:", best_acc)
plt.figure()
plt.plot(range(len(top_train_accuracy_list)), top_train_accuracy_list, label='Training Accuracy')
plt.plot(range(len(top_val_accuracy_list)), top_val_accuracy_list, label='Validation Accuracy')
plt.title('Training and Validation Accuracy for Best Hyperparameters')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()