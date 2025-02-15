# -*- coding: utf-8 -*-
"""
Code for producing the context embedding (LSTM output) from the regional LSTM model
The output is afterwards fed to a RF or other post-processing ML models
"""
#%%
#Import the necessary libraries
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import sys
import os
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





seq_length = 365 #input size
batch_size = int(sys.argv[1]) #read from bash file


# Extract feature column names, excluding 'q' (target variable) and 'basin_id' (identifier)
columns = df_tr.columns.to_list()
columns.remove('q')
columns.remove('basin_id')




class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        out = self.dropout(lstm_out[:, -1, :]) # take the last time step output
        # out shape: (batch_size, hidden_size)
        out = self.linear(out)
        # out shape: (batch_size, output_size)
        return out
        
class ModifiedLSTMModel(nn.Module):
    def __init__(self, original_model):
        """
        Wrap the original LSTM model to output a specific layer's hidden state.

        Args:
            original_model (nn.Module): The pretrained LSTM model.
        """
        super(ModifiedLSTMModel, self).__init__()
        self.lstm = original_model.lstm  # LSTM layer
    def forward(self, x, i=1):
        """
        Modify forward to output the hidden state of the last `-i` layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).
            i (int): The index for the last `-i` layer (1 = last layer, 2 = second-to-last, etc.).

        Returns:
            torch.Tensor: Hidden state of the specified layer of shape (batch_size, hidden_size).
        """
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        return lstm_out[:, -1, :]  # Return the specified layer's hidden state (batch_size, hidden_size)



# Define model hyperparameters
input_size = 32  # Number of input features (dynamic + static)
hidden_size = int(sys.argv[2])  # Hidden layer size (passed as command-line argument)
num_layers = 1  # Number of LSTM layers
output_size = 1  # Single output (streamflow prediction)
dropout_prob = float(sys.argv[3])  # Dropout probability for regularization
seed = int(sys.argv[5])  # Random seed for reproducibility

# Initialize the LSTM model with the specified parameters
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)

# Load pre-trained weights for the model
model.load_state_dict(torch.load(f'ModelLSTM_st_random_Final_seed_{seed}', weights_only=True))

# Move the model to the appropriate device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set the model to evaluation mode (disables dropout, batch norm updates)

# Create an encoded version of the model using a modified architecture
model_encode = ModifiedLSTMModel(model)
model_encode.to(device)

# Delete the original model to free up memory
del model


# Define the file path to save the results
save_results = f'./EncodedLSTM_st_random_{seed}'

# Iterate over all 421 catchments
for ii in range(0, 421):
  
    # Extract training data for the current basin (basin_id == ii)
    temp_x = np.asarray(df_tr[df_tr['basin_id'] == ii].loc[:, columns])
    temp_y = np.asarray(df_tr[df_tr['basin_id'] == ii]['q']).reshape((-1, 1))
    x_, y_ = split_sequence_multi_train(temp_x, temp_y, 365, 0, mode='seq')

    # Convert training data to PyTorch tensors
    X_train = torch.tensor(x_, dtype=torch.float32)

    # Extract test data for the current basin (basin_id == ii)
    temp_xx = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii].loc[:, columns])
    temp_yy = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii]['q']).reshape((-1, 1))
    xx_, yy_ = split_sequence_multi_train(temp_xx, temp_yy, 365, 0, mode='seq')

    # Convert test data to PyTorch tensors
    X_test = torch.tensor(xx_, dtype=torch.float32)
    
    # Set the model to evaluation mode (disables dropout, batch norm updates)
    model_encode.eval()

    # Move test data to the appropriate device (GPU or CPU)
    test_X = X_test.to(device).float()
  
  
  
    # Disable gradient calculation since we're in evaluation mode
    with torch.no_grad():
        for jj in range(0, len(X_train), batch_size):
            X_train_temp=X_train[jj:jj + batch_size]
            train_X = X_train_temp.to(device).float()
            # Get model predictions on the test data
            y_pred_train = model_encode(train_X)
            # Move predictions and true labels back to CPU for further processing
            if jj==0:
                y_pred_train_all = y_pred_train.cpu().numpy()
            else:
                y_pred_train_all=np.concatenate((y_pred_train_all,y_pred_train.cpu().numpy()),axis=0)
        
        y_pred_test = model_encode(test_X)
        # Move predictions back to CPU for further processing
        y_pred_test = y_pred_test.cpu().numpy()
        
        
    
    torch.cuda.empty_cache()
    
    # Create a temporary DataFrame for the current catchment
    df_temp_train = pd.DataFrame(y_pred_train_all)
    df_temp_test = pd.DataFrame(y_pred_test)
    
    del y_pred_train
    del y_pred_test
    
    try:
      df_temp_train.to_csv(save_results+'/'+'LSTM50_%d_Encoded_Train_%d.csv'%(hidden_size,ii),index=None)
      df_temp_test.to_csv(save_results+'/'+'LSTM50_%d_Encoded_Test_%d.csv'%(hidden_size,ii),index=None)
    except Exception as e: #If the folder does not exist
      print(e)
      print('Creating Directory')
      os.mkdir(save_results)
      df_temp_train.to_csv(save_results+'/'+'LSTM50_%d_Encoded_Train_%d.csv'%(hidden_size,ii),index=None)
      df_temp_test.to_csv(save_results+'/'+'LSTM50_%d_Encoded_Test_%d.csv'%(hidden_size,ii),index=None)
      

 
