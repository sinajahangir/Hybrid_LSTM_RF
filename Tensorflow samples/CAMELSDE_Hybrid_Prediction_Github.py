# -*- coding: utf-8 -*-
"""
First version: Oct 2024
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#This code is for obtaining predictions by the Hybrid (LSTM-Tree)
    ## The LSTM of the hybrid model was trained on 50 random catchments
    ## Fine-tuning was done by training the tree on the training portion
    
#important note: It seems it is not possible to load the saved DF models on some clusters.
    ##Google Colab can be an alternative options.
    ##We have to retrain the hybrid model and save the forecasts


#Tested on Python 3.11
Copyright (c) [2024] [Mohammad Sina Jahangir]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

#Dependencies:
-tf
-tfdf
-pandas
-numpy
"""
#%%
# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import time

# Load the training dataset and remove any rows with missing values
csv_path = 'All324data_Train_v1.csv'
df = pd.read_csv(csv_path)
df = df.dropna()

# Load the test dataset
csv_path_test = 'All324data_Test_v1.csv'
df_test = pd.read_csv(csv_path_test)

# Compute the mean and standard deviation for feature scaling (excluding the first column)
mean_ = np.asarray(df.iloc[:, 1:].mean())  # Compute column-wise mean
std_ = np.asarray(df.iloc[:, 1:].std())  # Compute column-wise standard deviation

# Standardize the test dataset using the training set statistics
df_test_tr = df_test.iloc[:, 1:] - mean_  # Subtract mean
df_test_tr = df_test_tr / std_  # Divide by standard deviation

# Ensure 'basin_id' is retained in the test dataset
df_test_tr = df_test_tr.drop(columns=['basin_id'])  # Drop 'basin_id' (if present)
df_test_tr['basin_id'] = df_test['basin_id']  # Reinsert 'basin_id' column

# Compute mean and standard deviation for the target variable (discharge)
mean_q = df['discharge_spec'].mean()  # Mean of discharge
std_q = df['discharge_spec'].std()  # Standard deviation of discharge
#%%
def split_sequence_multi_train(sequence_x,sequence_y, n_steps_in, n_steps_out,mode='seq'):
    """
    written by:SJ
    sequence_x=features; 2D array
    sequence_y=target; 2D array
    n_steps_in=IL(lookbak period);int
    n_steps_out=forecast horizon;int
    mode:either single (many to one) or seq (many to many).
    This function creates an output in shape of (sample,IL,feature) for x and
    (sample,n_steps_out) for y
    """
    X, y = list(), list()
    k=0
    sequence_x=np.copy(np.asarray(sequence_x))
    sequence_y=np.copy(np.asarray(sequence_y))
    for _ in range(len(sequence_x)):
		# find the end of this pattern
        end_ix = k + n_steps_in
        out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
        if out_end_ix > len(sequence_x):
            break
		# gather input and output parts of the pattern
        seq_x = sequence_x[k:end_ix]
        #mode single is used for one output
        if n_steps_out==0:
            seq_y= sequence_y[end_ix-1:out_end_ix]
        elif mode=='single':
            seq_y= sequence_y[out_end_ix-1]
        else:
            seq_y= sequence_y[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y.flatten())
        k=k+1

    XX,YY= np.asarray(X), np.asarray(y)
    if (n_steps_out==0 or n_steps_out==1):
        YY=YY.reshape((len(XX),1))
    return XX,YY
#%%
# Standardize the training dataset (excluding the first column)
# Each column is standardized by subtracting the mean and dividing by the standard deviation
df_tr = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Predefine lists for storing observations and predictions
# These lists will be populated in the loop below
diff_obs = []  # Stores observations for cases with different predictions
diff_pred = []  # Stores predictions for cases with differences

same_obs = []  # Stores observations for cases with identical predictions
same_pred = []  # Stores predictions for cases with no differences
#%%
# Loop over different random simulations (currently only one iteration)
for jj in range(0, 1):
    # Load catchment indices used for model training
    list_ = list(pd.read_csv(f'./PHIMP/random_numbers_sim_{jj}.csv').iloc[:, 1])
    
    # Define the entire range of catchments
    list_whole = np.arange(0, 324)

    # Define LSTM-based encoder model
    inputs = tf.keras.layers.Input(shape=(365, 9))
    x = tf.keras.layers.LSTM(64, return_sequences=False)(inputs)  # LSTM layer with 64 hidden units
    xx = tf.keras.layers.Dropout(0.2)(x)  # Apply dropout regularization
    outputs = tf.keras.layers.Dense(1, activation='linear')(xx)  # Linear output layer

    # Create and compile model
    modeli = tf.keras.Model(inputs, outputs)
    modeli.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3), loss="mse")

    # Load pre-trained LSTM weights
    modeli.load_weights(f"./Checkpoint/Final_sim64_{jj}.h5")

    # Extract LSTM encoder (excluding the final output layer)
    nn_without_head = tf.keras.models.Model(inputs=modeli.inputs, outputs=x)

    # Loop over all 324 catchments
    for ii in range(0, 324):
        # Extract training data for a specific catchment
        temp_x = np.asarray(df_tr[df['basin_id'] == ii].loc[:, [
            'precipitation_mean', 'precipitation_stdev', 'radiation_global_mean', 
            'temperature_min', 'temperature_max', 'average_pr', 'average_q', 
            'average_tmax', 'average_tmin'
        ]])
        temp_y = np.asarray(df_tr[df['basin_id'] == ii]['discharge_spec']).reshape((-1, 1))

        # Create training sequences
        xx, yy = split_sequence_multi_train(temp_x, temp_y, 365, 0, mode='seq')

        # Train Random Forest model with the LSTM-encoded features
        model = tfdf.keras.RandomForestModel(
            preprocessing=nn_without_head,
            task=tfdf.keras.Task.REGRESSION,
            maximum_training_duration_seconds=300.0  # Train for up to 300 seconds
        )

        # Measure training time
        st = time.time()
        model.fit(x=xx, y=yy[:, 0].reshape((-1, 1)))
        et = time.time()
        print(f"Training time for basin {ii}: {et - st:.2f} seconds")

        # Extract test data for the same catchment
        temp_xx = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii].loc[:, [
            'precipitation_mean', 'precipitation_stdev', 'radiation_global_mean', 
            'temperature_min', 'temperature_max', 'average_pr', 'average_q', 
            'average_tmax', 'average_tmin'
        ]])
        temp_yy = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii]['discharge_spec']).reshape((-1, 1))

        # Create test sequences
        xx_, yy_ = split_sequence_multi_train(temp_xx, temp_yy, 365, 0, mode='seq')

        # Ground truth streamflow values
        y_out_tr = yy_[:, 0]

        # Obtain predictions from the trained model
        y_pred_tr = model.predict(xx_)
        y_m_out_tr = y_pred_tr

        # Transform predictions and ground truth back to the original scale
        y_out_tr = y_out_tr * std_q + mean_q
        y_m_out_tr = y_m_out_tr * std_q + mean_q

        # Store observations and predictions in appropriate lists
        if ii in list_:
            same_obs.append(y_out_tr)
            same_pred.append(y_m_out_tr)
        else:
            diff_obs.append(y_out_tr)
            diff_pred.append(y_m_out_tr)

        # Delete model to free up memory
        del model

    
    # Identify catchments in list_whole that are not in list_ (i.e., those not used for training)
    list_rm = [item for item in list_whole if item not in list_]
    
    # Convert lists to pandas DataFrames and save them as CSV files
    
    # Observations from catchments used for training
    df_same_obs = pd.DataFrame(np.asarray(same_obs).T, columns=list_)
    df_same_obs.to_csv('CAMELSDE_Tree_Same_Obs.csv', index=False)
    
    # Observations from catchments NOT used for training
    df_diff_obs = pd.DataFrame(np.asarray(diff_obs).T, columns=list_rm)
    df_diff_obs.to_csv('CAMELSDE_Tree_Diff_Obs.csv', index=False)
    
    # Predictions for catchments used for training
    df_same_pred = pd.DataFrame(np.asarray(same_pred).T[0, :, :], columns=list_)
    df_same_pred.to_csv('CAMELSDE_Tree_Same_Pred.csv', index=False)
    
    # Predictions for catchments NOT used for training
    df_diff_pred = pd.DataFrame(np.asarray(diff_pred).T[0, :, :], columns=list_rm)
    df_diff_pred.to_csv('CAMELSDE_Tree_Diff_Pred.csv', index=False)