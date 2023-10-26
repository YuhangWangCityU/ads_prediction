#!/usr/bin/env python
# coding: utf-8

# # Import required modules

# In[2]:


import pandas as pd
import numpy as np 
import time
import matplotlib.pyplot as plt
import warnings
import os 
import string 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold   
from sklearn.model_selection import train_test_split 
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
from matplotlib.pyplot import MultipleLocator
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

warnings.filterwarnings("ignore")
start = time.time()


# In[3]:


# Read data
file_name='Feature_CO.xlsx'
df = pd.read_excel('../'+file_name, sheet_name= "ML_features")
features  = df.iloc[:,:-1]


# In[4]:


# standardize the descriptor 
mean1 = features.mean(axis=0)                              ## 求平均， numpy中的功能 
std1 = features.std(axis=0)                                #  Z标准化：实现中心化和正态分布 
# comment this line if you want unstandardized descriptors  
features = (features - mean1)/ std1                             #  Z标准化：实现中心化和正态分布  
# features
target = df.iloc[:,-1]


# # Code for Building and Training Neural Network Models

# In[6]:


# self.graph = tf.compat.v1.get_default_graph()
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
def mean_averaged_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))
def create_model(num_cols):
    # Initialising the ANN
    model = Sequential()
    tf.random.set_seed(7)
    # Adding the input layer and the first hidden layer
    model.add(Dense(30, kernel_initializer='uniform', activation='relu', input_dim=num_cols))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the second hidden layer
    model.add(Dense(15, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the third hidden layer
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(epsilon=1e-06, momentum=0.8, weights=None))

    # Adding the output layer
    model.add(Dense(1, kernel_initializer='uniform'))

    return model

def train_model_rmse(X_train, y_train, num_cols):
    # Create model
    model = create_model(num_cols)                  ### 神经网络的训练是需要搭建的
    tf.random.set_seed(7)

    # Define early stopping
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('model_best_weights.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    # Compiling the ANN
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=root_mean_squared_error, optimizer=optimizer)

    # Fitting the ANN to the Training set                           # 这里开始训练
    model_history = model.fit(X_train.values.astype('float32'), y_train.values.astype('float32'),
                              validation_split=0.1, batch_size=64, epochs=2000, callbacks=[checkpoint])
    return model_history

def train_model_mae(X_train, y_train, num_cols):
    # Create model
    model = create_model(num_cols)
    tf.random.set_seed(7)

    # Define early stopping
    # early_stop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, mode='min', verbose=1)
    checkpoint = ModelCheckpoint('model_best_weights_mae.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min', period=1)

    # Compiling the ANN
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(loss=mean_averaged_error, optimizer=optimizer)

    # Fitting the ANN to the Training set
    model_history = model.fit(X_train.values.astype('float32'), y_train.values.astype('float32'),
                              validation_split=0.1, batch_size=64, epochs=2000, callbacks=[checkpoint])

    return model_history

def load_model(weights_path, num_cols):
    model = create_model(num_cols)
    model.load_weights(weights_path)
    return model


# # Train NN using RMSE as the loss function

# In[7]:


# train model RMSE 
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state= 42)

start = time.time()
model_history_rmse = train_model_rmse(X_train, y_train, num_cols = len(features.columns))
end = time.time()

training_time = end - start 
print (training_time)


# In[9]:


# print RMSE
# nn_RMSE = min(model_history_rmse.history['val_loss'])
# nn_RMSE


# In[10]:


# print (model_history_rmse.history['val_loss'] )


# In[11]:


nn_RMSE_mean = np.mean(model_history_rmse.history['val_loss'][1000:])
# nn_RMSE_mean


# In[12]:


# train model MAE 
start = time.time()
model_history_mae = train_model_mae(X_train, y_train, num_cols = len(features.columns))
end = time.time()


# In[13]:


# Find the min MAE 
# nn_MAE = min(model_history_mae.history['val_loss'])
# nn_MAE

nn_MAE_mean = np.mean(model_history_mae.history['val_loss'][1000:])
nn_MAE_mean


# # Training/Validation Curves 

# In[14]:


# For MAE
font = {'size': '28'}
plt.rc('font', **font)

training_losses = model_history_mae.history['loss']
validation_losses = model_history_mae.history['val_loss']

fig, ax = plt.subplots(figsize=(8.75, 6.5))
plt.plot(training_losses, label='Training Loss', linewidth =2.5)
plt.plot(validation_losses, label='Validation Loss', linewidth = 2.5, c='g')

ax.set_xlabel('Epoch')
ax.set_ylabel('MAE (eV)')

# ax.tick_params(axis='x', colors='black', labelsize = 24, width=6, length=15, color='black')
# ax.tick_params(axis='y', colors='black', labelsize= 24,  width=6, length=15, color='black')

ax.tick_params(axis='x', labelsize = 24,)
ax.tick_params(axis='y', labelsize= 24, )

plt.legend(loc=0, fontsize=24)    
    
# plt.savefig('MAE_Loss.jpeg', dpi=600, bbox_inches='tight')
plt.show()


# In[15]:


# For RMSE
font = {'size': '28'}
plt.rc('font', **font)

training_losses = model_history_rmse.history['loss']
validation_losses = model_history_rmse.history['val_loss']

fig, ax = plt.subplots(figsize=(8.75, 6.5))
plt.plot(training_losses, label='Training Loss', linewidth =2.5)
plt.plot(validation_losses, label='Validation Loss', linewidth = 2.5, c='g')

ax.set_xlabel('Epoch',)
ax.set_ylabel('RMSE (eV)', )

# ax.tick_params(axis='x', colors='black', labelsize = 24, width=6, length=15, color='black')
# ax.tick_params(axis='y', colors='black', labelsize= 24,  width=6, length=15, color='black')

ax.tick_params(axis='x', labelsize = 24,)
ax.tick_params(axis='y', labelsize= 24, )

plt.legend(loc=0, fontsize=24)      
    
# plt.savefig('RMSE_loss.jpeg', dpi=600, bbox_inches='tight')
plt.show()


# In[16]:


# obtain optimized NN model#  Model performance after hyperparameter tuning
nn = load_model('model_best_weights.h5', num_cols = len(features.columns))


# #  Model performance after training

# In[32]:


# data splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=47) 

# Scatterplot of predicted versus actual values
font = {'size': '24'}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(8.67, 6.5))
ax.scatter(y_train, y_tr_pred, alpha=0.5, c='#4DBBD5', marker='o', label='Training', s=250)
ax.scatter(y_test, y_te_pred, alpha=0.5, c='#f67280', marker='o', label='Test', s=250)

plt.xlim(-3, 0)
plt.ylim(-3, 0)

ax.set_xlabel('DFT (eV)')
ax.set_ylabel('Prediction (eV)')

xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, 
        scalex=False, scaley=False)


y_major_locator=MultipleLocator(0.5)
ax.yaxis.set_major_locator(y_major_locator)
x_major_locator=MultipleLocator(0.5)
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 


plt.grid(axis='x', ls='--', alpha=0.3)
plt.grid(axis='y', ls='--', alpha=0.3)

plt.text(0.5, 0.3, 'NN Test', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('NN')
# plt.savefig('Model_performance_opt_para-0405.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # 500 times training/test

# In[ ]:


print ('---500 times running ---')
R2_2nd = []
RMSE_2nd = []

R2_2nd_test = []
RMSE_2nd_test = []

for i in range(0,500):
    print (i)
    X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8) 

    y_tr_pred = model_optimized.predict(X_train)
    rmse_scores = np.sqrt(mean_squared_error(y_tr_pred, y_train))
    RMSE_2nd.append(rmse_scores)
    print ('RMSE for training (v2): {:.4f} eV'.format(np.sqrt(mean_squared_error(y_tr_pred, y_train))))
    
    R_squr_train = r2_score(y_train, y_tr_pred)
    print ('R^2 for train {:.4f} eV'.format(R_squr_train))


    y_tr_pred = model_optimized.predict(X_train)
    R_squr_train = r2_score(y_train, y_tr_pred)
    print ('R^2 for train {:.4f} '.format(R_squr_train))
    R2_2nd.append(R_squr_train)
    
    # 开始做测试
    y_te_pred = model_optimized.predict(X_test)
    MSE = mean_squared_error(y_te_pred, y_test)
    RMSE = MSE**.5
    
    RMSE_2nd_test.append(RMSE)
    
    R_squr = r2_score(y_test, y_te_pred)
    R2_2nd_test.append(R_squr)

test1 = pd.DataFrame({'nn-rmse':RMSE_2nd})
test2 = pd.DataFrame({'nn-r2':R2_2nd})
test3 = pd.DataFrame({'nn-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'nn-r2_test':R2_2nd_test})

# Creating Excel Writer Object from Pandas  
writer = pd.ExcelWriter('model_performance.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='nn1',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='nn2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='nn3',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='nn4',startrow=0, startcol=0)
writer.close()    


# In[ ]:




