import FFNN_BLACKBOX

#Here we will use the test data set using the TOP hyperparameters obtained through the grid search.
batch_size = 184

#loading data
train_features, train_labels = FFNN_BLACKBOX.Load_Data('DR_train_data.csv')
test_features, test_labels = FFNN_BLACKBOX.Load_Data('DR_test_data.csv')
#we treat the test features as the validation set now.
dummy_features, dummy_labels =FFNN_BLACKBOX.Load_Data('DR_test_data.csv') #never used

#create loaders
train_loader, test_loader, dummy_loader = FFNN_BLACKBOX.create_dataloader(train_features, train_labels, test_features, test_labels, dummy_features,dummy_labels, 184)

#again, just plug in the results from the hyperparameter Randomsearch
LR =0.013494565585173276
input_features = train_features.shape[1]
hidden_neurons = 201
n_layers =1
output_neurons = 10
activation_function='relu'
epochs =102
model, optimizer, criterion =FFNN_BLACKBOX.initialize_network(input_features, hidden_neurons, n_layers, output_neurons, activation_function, LR=LR)
FFNN_BLACKBOX.test_set_eval(train_loader, test_loader, model, optimizer, criterion, epochs)
