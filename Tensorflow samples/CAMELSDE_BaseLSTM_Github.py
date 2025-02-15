# -*- coding: utf-8 -*-
"""
@author: Mohammad Sina Jahangir (Ph.D.)
Email:mohammadsina.jahangir@gmail.com
#The developed code is for training the base LSTM model on 50 random cathcments selected from CAMELS-DE
    
#important note: It seems it is not possible to load the DF models on HPC.
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
-pandas
-numpy
"""
#import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
#%%
# Load the training dataset and remove any rows with missing values
csv_path='All324data_Train_v1.csv'
df = pd.read_csv(csv_path)
df=df.dropna()
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
# Standardize the training dataset (excluding the first column)
# Each column is standardized by subtracting the mean and dividing by the standard deviation
df_tr = df.iloc[:, 1:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
#%%
#custom MSE
def custom_loss(y_true, y_pred):
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true))) + 1e-5
    nse = numerator / denominator -1
    return nse
#%%
#Base LSTM model
def modeli_():
    inputs = tf.keras.layers.Input(shape=(365, 9))
    x = tf.keras.layers.LSTM(64, return_sequences=False)(inputs)
    x=tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1,activation='linear')(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-4), loss=custom_loss)
    return model
#%%
#Define clabback for avoiding overfitting 
reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=10, min_lr=1e-6)
early_callback=tf.keras.callbacks.EarlyStopping(patience=30, monitor='val_loss', restore_best_weights=True)
for kk in range (0,1):
    # Set random seed
    random.seed(kk+1)
    # Generate 50 random integers
    random_numbers = random.sample(range(324), 50)
    # Write  random numbers to a CSV file for future retrieval
    df_random=pd.DataFrame(random_numbers).to_csv('./PHIMP/random_numbers_sim64_%d.csv'%(kk))
    model=modeli_()
    k=0
    for ii in random_numbers:

      temp_x=np.asarray(df_tr[df['basin_id']==ii].loc[:, ['precipitation_mean', 'precipitation_stdev',
       'radiation_global_mean', 'temperature_min', 'temperature_max', 'average_pr',
           'average_q', 'average_tmax', 'average_tmin']])
      temp_y=np.asarray(df_tr[df['basin_id']==ii]['discharge_spec']).reshape((-1,1))
      xx,yy=split_sequence_multi_train(temp_x,temp_y,365,0,mode='seq')
      if k==0:
        x_train=xx[int(0.1*len(xx)):int(0.9*len(xx))]
        y_train=yy[int(0.1*len(xx)):int(0.9*len(xx))]

        x_val=xx[:int(0.1*len(xx))]
        y_val=yy[:int(0.1*len(xx))]

      else:
        x_train=np.concatenate((x_train,xx[int(0.1*len(xx)):int(0.9*len(xx))]),axis=0)
        y_train=np.concatenate((y_train,yy[int(0.1*len(xx)):int(0.9*len(xx))]),axis=0)


        x_val=np.concatenate((x_val,xx[:int(0.1*len(xx))]),axis=0)
        y_val=np.concatenate((y_val,yy[:int(0.1*len(xx))]),axis=0)
      k=k+1
      
    with tf.device("CPU"):

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=1024).batch(32)
        dataset_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).shuffle(buffer_size=1024).batch(32)

    del xx, yy,x_train,y_train,x_val,y_val
    

    # Train the model using the generator
    # Include the epoch in the file name (uses `str.format`)
    checkpoint_path = "./Checkpoint/sim64_%d_{epoch:03d}.h5"%(kk)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch')

    st = time.time()
    history = model.fit(dataset, epochs=300,validation_data=dataset_val, callbacks=[early_callback,cp_callback,reduce_callback],batch_size=32,verbose=2)
    et= time.time()
    #save the training time
    df_time=pd.DataFrame([et-st],columns=['Time (s)'])
    df_time.to_csv('./PHIMP/times_sim64_%d.csv'%(kk),index=False)
    #save model weights
    model.save_weights("./Checkpoint/Final_sim64_%d.h5"%(kk))
    del model
    del dataset
    del dataset_val