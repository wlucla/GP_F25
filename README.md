# GP_F25
CURRENT BEST HYPERPARAMETERS FOR OUR FFNN:
Best Hyperparameters: {'learning_rate': 0.013494565585173276, 'hidden_neurons': 201, 'n_layers': 1, 'activation_function': 'relu', 'batch_size': 184}
Best TEST Accuracy : 0.9453333333333334

PERFORMANCE OF CNN FROM MIT LICENCE: 98.8% test accuracy after 1 epoch only. Batch size =184.


CONCLUSION: FFNN underperforms CNN in both accuracy and computational complexity.

STEPS TO RUN THE ALGORITHM COMPLETELY.

0. Clone repository.

1. If not already, delete intermediate_files
2. If not already, delete DR_test_data.csv, DR_train_data.csv, DR_val_dataset.csv
3. RUN main1_dataPreprocess.py (create PCA processed data split into test, train, val)
4. RUN mainHyperparameterRandomSearch.py ; in line 13, edit the number of random sets of hyperparameters you want to test. At 100 sets, training can take really wrong (~2.3 hours). Default to 100 sets.
5. RUN MAIN_TEST_FFNN.py with the best hyperparameters found in step 4. (yields test accuracy)
6. RUN the MIT notebok cnn.ipynb and compare our FFNN and their CNN's performance on the same task. (make sure batch size is the same as ours)






Extra notes(Not really importantant, since its included in detail in the report)

PCA implementation to reduce to n principle components: takes input 784x1 flattened image and processes it into a nx1 feature vector before any training happens. Implementation complete. 

***IF THE ONLY .CSV FILE YOU SEE IS digits_dataset.csv, THEN RUN main1_dataPreprocess.py ONCE. digits_dataset.csv IS THE RAW DATASET. DO NOT DELETE IT. ANY OTHER .CSV IS SECONDARY AND CAN BE DELETED (IF YOU DO DELETE ANY OF THE SECONDARY .CSVs, DELETE ALL SECONDARIES AND RUN main1_dataPreprocess.py again. This means delete the entire intermediate_files folder too)

***THIS WILL CREATE 3 FILES THAT YOU WILL USE FOR TRAINING, TESTING, VALIDATION: DR_test_data.csv, DR_train_data.csv, DR_val_data.csv. For each of these csv files, the column 0 is the true digit. The remaining 'n' principle components are the remaining columns.

If you dont care or know about what PCA is, just know that after you run main1_dataPreprocess.py, your train, val, and test datasets are DR_test_data.csv, DR_train_data.csv, DR_val_data.csv. Dont use the other csv files.

