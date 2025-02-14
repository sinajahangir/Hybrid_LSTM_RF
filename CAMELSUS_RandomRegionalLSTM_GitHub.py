# -*- coding: utf-8 -*-
"""
This project implements a regional LSTM model for streamflow prediction using the CAMELS-US dataset
The model is trained on 371 randomly selected catchments out of 421, optimized with MSE loss
and utilizes early stopping for regularization.
"""
#%%
#Import the necessary libraries
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import sys
import random
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
#%%
#Error metrics
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
#%%
# Torch data pipeline for window batching
from torch.utils.data import Dataset, DataLoader,ConcatDataset
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for handling large 2D arrays and converting them to LSTM-ready 3D sequences.
    """
    def __init__(self, data, targets, seq_length):
        """
        Args:
            data (np.ndarray or torch.Tensor): The 2D array of shape [num_samples, num_features].
            targets (np.ndarray or torch.Tensor): The 1D array of target values.
            seq_length (int): The length of the sequence for LSTM.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        sequence = self.data[idx: idx + self.seq_length,:]
        target = self.targets[idx + self.seq_length - 1]  # Align target with the end of the sequence
        return sequence, target
#%%
seq_length = 365 #input lag
batch_size = 1024

# Extract feature column names, excluding 'q' (target variable) and 'basin_id' (identifier)
columns = df_tr.columns.to_list()
columns.remove('q')
columns.remove('basin_id')

# Set random seed for reproducibility
seed = int(sys.argv[1])  # Seed provided as a command-line argument
torch.manual_seed(seed)  # PyTorch seed
torch.cuda.manual_seed_all(seed)  # Ensure CUDA consistency
np.random.seed(seed)  # NumPy seed
random.seed(seed)  # Python's built-in random module
torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior in CUDA

# Generate 371 random basin IDs from a total of 421 (exclude 50)
random_numbers = random.sample(range(421), 371)

# Save the random numbers to a CSV file for reproducibility
df_random = pd.DataFrame(random_numbers)
df_random.to_csv(f'random_numbers_reg_sim_{seed}.csv', index=False)

# Initialize dataset variables
kk = 0
# Iterate through selected random basin IDs
for ii in random_numbers:
    # Extract input features and target values for the selected basin
    data_all = np.asarray(df_tr[df['basin_id'] == ii].loc[:, columns])
    targets = np.asarray(df_tr[df['basin_id'] == ii]['q']).reshape((-1, 1))

    # Split into training (90%) and validation (10%) sets
    dataset_temp = TimeSeriesDataset(
        data_all[:int(0.9 * len(data_all)), :], 
        targets[:int(0.9 * len(data_all)), :].reshape((-1, 1)), 
        seq_length
    )
    dataset_val_temp = TimeSeriesDataset(
        data_all[int(0.9 * len(data_all)):, :], 
        targets[int(0.9 * len(data_all)):, :].reshape((-1, 1)), 
        seq_length
    )

    # Concatenate datasets
    if kk == 0:
        dataset = dataset_temp
        dataset_val = dataset_val_temp
    else:
        dataset = ConcatDataset([dataset, dataset_temp])
        dataset_val = ConcatDataset([dataset_val, dataset_val_temp])

    kk += 1  # Increment counter

# Create PyTorch DataLoaders for training and validation
dataloader_train = DataLoader(dataset, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
#%%
#LSTM model
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
#%%
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

# Model parameters
input_size = 32  # Dynamic+static
hidden_size = 256
num_layers = 1
output_size = 1
dropout_prob = 0.4

# Initialize the model
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_epochs_trained = 40
patience = 4
best_val_loss = float('inf')
early_stop_counter = 0
best_model_state = None
epochs_trained = 0
optimizer = optim.Adam(model.parameters())
scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.1, min_lr=1e-6)
loss_fn =nn.MSELoss()
for epoch in range(max_epochs_trained):
    print('new epoch')
    print(epoch)
    model.train()
    epoch_loss = 0.0
    for X_batch, Y_batch in dataloader_train:
        # Move data to the same device as the model
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, Y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Compute average training loss
    epoch_loss /= len(dataloader_train.dataset)

    # Validation every 10 epochs
    if epoch % 2 == 0:
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for XX_batch, YY_batch in dataloader_val:
                # Move data to the same device as the model
                XX_batch, YY_batch = XX_batch.to(device), YY_batch.to(device)

                # Forward pass for validation
                y_pred = model(XX_batch)
                loss = loss_fn(y_pred, YY_batch)
                val_loss += loss.item()

        # Compute average validation loss
        val_loss /= len(dataloader_val.dataset)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_state = model.state_dict()
            #torch.save(best_model_state, 'ModelLSTM_st_hyper%d_seed_%d'%(epoch,seed))
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}!")
                break

        print(f"Epoch {epoch}: Train Loss: {epoch_loss:.5f}, Val Loss: {val_loss:.5f}")
        scheduler.step(val_loss)
        print(scheduler.get_last_lr())
    epochs_trained += 1
    # Step the scheduler


    if epochs_trained >= max_epochs_trained:
        print(f"Reached maximum of {max_epochs_trained} training epochs.")
        break

# Restore the best model
if best_model_state:
    model.load_state_dict(best_model_state)
    print("Best model restored!")
torch.save(best_model_state, 'ModelLSTM_st_random_Final_seed_%d'%(seed))
del model
#Make sure to load the model to check if everything is ok
modeli=LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob)
modeli.to(device)
modeli.load_state_dict(torch.load('ModelLSTM_st_random_Final_seed_%d'%(seed),weights_only=True))
#%%
# Iterate over all 421 catchments (both trained and non-trained)
for ii in range(0, 421):
    # Extract test features and target values for the given basin ID
    temp_xx = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii].loc[:, columns])
    temp_yy = np.asarray(df_test_tr[df_test_tr['basin_id'] == ii]['q']).reshape((-1, 1))

    # Prepare input sequences for the model using a custom function
    xx_, yy_ = split_sequence_multi_train(temp_xx, temp_yy, 365, 0, mode='seq')

    # Convert NumPy arrays to PyTorch tensors
    X_test = torch.tensor(xx_, dtype=torch.float32).to(device)
    Y_test = torch.tensor(yy_, dtype=torch.float32).to(device)

    # Set model to evaluation mode (disables dropout and batch normalization layers)
    modeli.eval()

    # Disable gradient calculation for faster inference
    with torch.no_grad():
        # Get model predictions on the test data
        y_pred = modeli(X_test)

        # Move predictions and true labels back to CPU for further processing
        y_pred = y_pred.cpu().numpy()
        y_test = Y_test.cpu().numpy()

    # Convert normalized predictions and ground truth back to original scale
    y_test = yy_ * std_q + mean_q
    pred_test = y_pred * std_q + mean_q

    # Compute evaluation metrics
    nse = nash_sutcliffe_error(y_test, pred_test)  # Nash-Sutcliffe Efficiency
    kge = KGE(pred_test, y_test)  # Kling-Gupta Efficiency

    # Free GPU memory
    torch.cuda.empty_cache()

    # Create a temporary DataFrame for the current basin
    df_temp = pd.DataFrame({'basin_id': [ii], 'nse': [nse], 'kge': [kge]})

    # Save results to CSV
    csv_filename = f'NSE_KGE_LSTM_st_random_seed_{seed}.csv'
    
    if ii == 0:
        # Save the first result, overwriting any existing file
        df_temp.to_csv(csv_filename, index=False)
    else:
        # Append new results to the existing file
        df_results = pd.read_csv(csv_filename)
        df_results = pd.concat([df_results, df_temp], ignore_index=True)
        df_results.to_csv(csv_filename, index=False)



