# -*- coding: utf-8 -*-
"""
This project implements a hybrid LSTM-RF model for streamflow prediction using the CAMELS-US dataset
The model is trained on 371 randomly selected catchments out of 421, optimized with MSE loss
and utilizes early stopping for regularization.
The predictions are made on all catchments
"""
#import necessary libraries
import numpy as np
import pandas as pd
#tfdf for tensor-flow decision making
import tensorflow_decision_forests as tfdf
import sys
#%%
# Define file paths for training and test datasets
train_csv_path = 'All421data_Train_st.csv'
test_csv_path = 'All421data_Test_st.csv'

# Load training data and remove missing values
df = pd.read_csv(train_csv_path).dropna()

# Load test data
df_test = pd.read_csv(test_csv_path)

# Compute mean and standard deviation for normalization (excluding the first column)
mean_ = np.asarray(df.iloc[:, 1:].mean())
std_ = np.asarray(df.iloc[:, 1:].std())

# Normalize test data using training set statistics
df_test_tr = df_test.iloc[:, 1:] - mean_  # Subtract mean
df_test_tr = df_test_tr / std_  # Divide by standard deviation

# Move 'basin_id' column to the end after transformation
df_test_tr = df_test_tr.drop(columns=['basin_id'])
df_test_tr['basin_id'] = df_test['basin_id']

# Compute mean and standard deviation for target variable 'q' in training data
mean_q = df['q'].mean()
std_q = df['q'].std()

# Normalize training data column-wise
df_tr = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Move 'basin_id' column to the end after transformation
df_tr = df_tr.drop(columns=['basin_id'])
df_tr['basin_id'] = df['basin_id']
#%%
#Window batching for LSTM model
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


# prompt: write nash-sutcliffe error index
def nash_sutcliffe_error(Q_obs,Q_sim):
    """
    Written by: SJ
    Q_obs: observed discharge; 1D vector
    Q_sim: simulated discharge; 1D vector
    This function calculates the NSE between observed and simulated discharges
    returns: NSE; float
    """
    if len(Q_sim)!=len(Q_obs):
        print('Length of simulated and observed discharges do not match')
        return
    else:
        num=np.sum(np.square(Q_sim-Q_obs))
        den=np.sum(np.square(Q_obs-np.mean(Q_obs)))
        NSE=1-(num/den)
        return NSE

def CC(Pr,Y):
    from scipy import stats
    Pr=np.reshape(Pr,(-1,1))
    Y=np.reshape(Y,(-1,1))
    return stats.pearsonr(Pr.flatten(),Y.flatten())[0]
def KGE(prediction,observation):

    nas = np.logical_or(np.isnan(prediction), np.isnan(observation))
    pred=np.copy(np.reshape(prediction,(-1,1)))
    obs=np.copy(np.reshape(observation,(-1,1)))
    r=CC(pred[~nas],obs[~nas])
    beta=np.nanmean(pred)/np.nanmean(obs)
    gamma=(np.nanstd(pred)/np.nanstd(obs))/beta
    kge=1-((r-1)**2+(beta-1)**2+(gamma-1)**2)**0.5
    return kge


# Extract feature column names, excluding 'q' (target variable) and 'basin_id' (identifier)
columns = df_tr.columns.to_list()
columns.remove('q')
columns.remove('basin_id')

#re-transform to the original space
def i_normalize(x_tr,mean_,sd_):
  """
  Written by:SJ
  mean_,sd_,:mean_, and sd_ used for initial transformation; 2D array
  x_tr:transformed input
  This function inverses the transformation
  returns:inverse transfomed
  """
  x_i=x_tr*sd_+mean_
  return x_i


# Parameters
hidden_size = int(sys.argv[2])
epoch_selected = int(sys.argv[4])
seed = int(sys.argv[5])

# Define the path to save encoded results
save_results = f'./EncodedLSTM_st_random_{seed}'

# Loop over all 421 catchments (basins)
for ii in range(0, 421):
    
    # Extract and prepare training data for the current basin
    temp_x = np.asarray(df_tr[df_tr['basin_id'] == ii].loc[:, columns])
    temp_y = np.asarray(df_tr[df_tr['basin_id'] == ii]['q']).reshape((-1, 1))
    x_, y_ = split_sequence_multi_train(temp_x, temp_y, 365, 0, mode='seq')
    
    # Load the encoded training data for the current basin from CSV
    df_enc = pd.read_csv(f'{save_results}/LSTM50_{hidden_size}_Encoded_Train_{ii}.csv', index_col=False)
    df_enc['q'] = y_[:, 0]
    
    # Convert to TensorFlow dataset and initialize Random Forest model
    dataset_rf = tfdf.keras.pd_dataframe_to_tf_dataset(df_enc, label="q", task=tfdf.keras.Task.REGRESSION)
    model_rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION, maximum_training_duration_seconds=200)
    
    # Train the Random Forest model
    model_rf.fit(x=dataset_rf)
    
    # Extract and prepare test data for the current basin
    temp_xx = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii].loc[:, columns])
    temp_yy = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii]['q']).reshape((-1, 1))
    xx_, yy_ = split_sequence_multi_train(temp_xx, temp_yy, 365, 0, mode='seq')
    
    # Load the encoded test data for the current basin from CSV
    df_enc_test = pd.read_csv(f'{save_results}/LSTM50_{hidden_size}_Encoded_Test_{ii}.csv', index_col=False)
    df_enc_test['q'] = yy_[:, 0]
    
    # Convert to TensorFlow dataset and predict using the trained Random Forest model
    dataset_rf_test = tfdf.keras.pd_dataframe_to_tf_dataset(df_enc_test, label="q", task=tfdf.keras.Task.REGRESSION)
    y_pred = model_rf.predict(dataset_rf_test)
    
    # Convert predictions and observed values to numpy arrays
    y_pred = np.asarray(y_pred).reshape((-1, 1))
    y_obs = yy_[:, 0].reshape((-1, 1))
    
    # Normalize predictions and observations
    y_pred = i_normalize(y_pred, mean_q, std_q)
    y_obs = i_normalize(y_obs, mean_q, std_q)
    
    # Initialize DataFrame to store results if it's the first basin
    if ii == 0:
        df_save = pd.DataFrame(columns=['obs', 'pred', 'basin_id'])
        
        # Store observed and predicted values
        df_save['obs'] = y_obs.ravel()
        df_save['pred'] = y_pred.ravel()
        df_save['basin_id'] = ii
        df_save.to_csv(f'LSTM421_rf_Pred_random_{seed}.csv', index=False)
    else:
        # For subsequent basins, read existing files and append results
        df_save = pd.read_csv(f'LSTM421_rf_Pred_random_{seed}.csv')
        df_save_temp = pd.DataFrame(columns=['obs', 'pred', 'basin_id'])
        df_save_temp['obs'] = y_obs.ravel()
        df_save_temp['pred'] = y_pred.ravel()
        df_save_temp['basin_id'] = ii
        df_save = pd.concat([df_save, df_save_temp], axis=0)
        df_save.to_csv(f'LSTM421_rf_Pred_random_{seed}.csv', index=False)
