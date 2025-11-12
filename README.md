# GP_F25
PCA implementation to reduce to n principle components: takes input 784x1 flattened image and processes it into a nx1 feature vector before any training happens. Implementation complete. 

***IF THE ONLY .CSV FILE YOU SEE IS digits_dataset.csv, THEN RUN main1_dataPreprocess.py ONCE. digits_dataset.csv IS THE RAW DATASET. DO NOT DELETE IT. ANY OTHER .CSV IS SECONDARY AND CAN BE DELETED (IF YOU DO DELETE ANY OF THE SECONDARY .CSVs, DELETE ALL SECONDARIES AND RUN main1_dataPreprocess.py again. This means delete the entire intermediate_files folder too)

***THIS WILL CREATE 3 FILES THAT YOU WILL USE FOR TRAINING, TESTING, VALIDATION: DR_test_data.csv, DR_train_data.csv, DR_val_data.csv. For each of these csv files, the column 0 is the true digit. The remaining 'n' principle components are the remaining columns.

If you dont care or know about what PCA is, just know that after you run main1_dataPreprocess.py, your train, val, and test datasets are DR_test_data.csv, DR_train_data.csv, DR_val_data.csv. Dont use the other csv files.



CURRENT BEST HYPERPARAMETERS:
Best Hyperparameters: {'learning_rate': 0.013494565585173276, 'hidden_neurons': 201, 'n_layers': 1, 'activation_function': 'relu', 'batch_size': 184}
Best Validation Accuracy: 0.9613333333333334