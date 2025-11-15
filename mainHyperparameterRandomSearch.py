# import hyperparameter_random_search
from hyperparameter_random_search import hyperparameter_random_search
import FFNN_BLACKBOX as ffnn
import matplotlib.pyplot as plt





train_features, train_labels = ffnn.Load_Data('DR_train_data.csv')
val_features, val_labels = ffnn.Load_Data('DR_val_data.csv')
test_features, test_labels = ffnn.Load_Data('DR_test_data.csv')
best_params, best_acc, top_train_accuracy_list, top_val_accuracy_list = hyperparameter_random_search(train_features, train_labels,val_features, val_labels,test_features, test_labels, trials=1)
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