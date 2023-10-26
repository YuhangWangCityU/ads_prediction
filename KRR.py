#!/usr/bin/env python
# coding: utf-8

# # Import required modules

# In[1]:


import pandas as pd
import numpy as np 
import time
import warnings
import os 
import string 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score      # RMSE
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
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
warnings.filterwarnings("ignore")
start = time.time()


# In[2]:


# Instantiate regressor algorithms
GBR = GradientBoostingRegressor(random_state=42)
KNR = KNeighborsRegressor()
RFR = RandomForestRegressor(random_state=42)
SR = SVR()
KRR = KernelRidge()
xgbr = XGBRegressor(random_state=42)
light = LGBMRegressor(random_state=42)


# In[3]:


# Read data
file_name='Feature_CO.xlsx'
df = pd.read_excel('../'+file_name, sheet_name= "ML_features")
features = df.iloc[:,:-1]
# features
target = df.iloc[:,-1]
# target


# In[5]:


# Adding gaussian noise to target values to prevent overfitting
np.random.seed(1)
mu  =  0    # 高斯噪声的均值
sigma = 0.065  # 高斯噪声的方差, 1. 0.065; 2.0.1
# [1, 106], 106是数据的行数
noise = np.random.normal(mu, sigma, [1,108]).tolist()
target_2 = target + noise[0]


# # Feature normalization

# In[6]:


# standardize the descriptor 
mean1 = features.mean(axis=0)                              ## 求平均， numpy中的功能 
std1 = features.std(axis=0)                                #  Z标准化：实现中心化和正态分布 
# comment this line if you want unstandardized descriptors  
features = (features - mean1)/ std1                             #  Z标准化：实现中心化和正态分布  


# # Model performance for default parameters

# In[12]:


#  data splitting  
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=42) 

# Define the model
model = KRR


kfold = KFold(n_splits=5, shuffle=True, random_state=10)
scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

# model scores
mse_ads = [abs(s) for s in scores]
mse_tr = np.mean(mse_ads)

model.fit(X_train, y_train)
y_tr_pred = model.predict(X_train)

# print ('----------------- RMSE -----------------')
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print ('RMSE for training (v2): {:.4f} eV'.format(np.sqrt(mean_squared_error(y_tr_pred, y_train))))

R_squr_train = r2_score(y_train, y_tr_pred)
print ('R^2 for train {:.4f} eV'.format(R_squr_train))

# model testing
y_te_pred = model.predict(X_test)


MAE = mean_absolute_error(y_te_pred, y_test)
MSE = mean_squared_error(y_te_pred, y_test)
RMSE = MSE**.5
print ('calcultae average RMSE for test {:.4f} eV'.format(RMSE))
R_squr = r2_score(y_test, y_te_pred)
print ('R^2 for test {:.4f} eV'.format(R_squr))

# Scatterplot of predicted versus actual values  
font = {'size': '28'}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(8.67, 6.5))
ax.scatter(y_train, y_tr_pred, alpha=0.5, c='b', marker='o', label='Training')
ax.scatter(y_test, y_te_pred, alpha=0.5, c='r', marker='o', label='Test')

plt.xlim(-3, 3)
plt.ylim(-3, 3)

ax.set_xlabel('DFT (eV)')
ax.set_ylabel('Prediction (eV)')

xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, 
        scalex=False, scaley=False)

plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes)

plt.legend(loc=0)
plt.title('KRR')
# plt.savefig('Model_performance_default_para.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # Hyperparameter tuning 

# In[15]:


n_folds = 5
def rmsle_cv(model, X_train, y_train):
    kf = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# In[16]:


def tuning(X_train, y_train):
    # range determined by the user
    degree = np.arange(2,6,1)
    alpha = np.arange(0,1.1,0.1)
    coef0 = np.arange(1,2.6,0.1)
    krr_score = []
    alpha_ = []
    coef0_ = []
    degree_ = []
    for i in range(0,len(degree)):
        for j in range(0,len(alpha)):
            for m in range(0,len(coef0)):
                model_optimized = KernelRidge(alpha = alpha[j], 
                                       coef0 = coef0[m], 
                                       degree = degree[i], 
                                       kernel = 'polynomial')           
                score = rmsle_cv(model_optimized, X_train, np.ravel(y_train))
                krr_score.append(score.mean())

                alpha_.append(alpha[j])
                coef0_.append(coef0[m])
                degree_.append(degree[i])
    krr_score_df = pd.DataFrame({'alpha': alpha_,'coef0': coef0_,'degree': degree_,'score': krr_score})
    krr_score_df = krr_score_df.sort_values(['score'], ascending = True)
#     krr_score_df.to_excel()
    return krr_score_df 


# In[17]:


krr_score_df = tuning(X_train, y_train)
print (krr_score_df)
model_optimized = KernelRidge(alpha = krr_score_df['alpha'].iloc[0],
                    coef0 = krr_score_df['coef0'].iloc[0],
                    degree = krr_score_df['degree'].iloc[0],
                    kernel = 'polynomial')

model_optimized.get_params
# model_optimized = model_grid_cv.best_estimator_
print (model_optimized)


# In[23]:


print (model)
print (model_optimized)


# #  Model performance after hyperparameter tuning

# In[28]:


# 重新定义数据的分割情况, 主要是调整 random_state
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=47) 
# df
# 定义模型
# RFR =  RandomForestRegressor() 
kfold = KFold(n_splits=10, shuffle=False)
scores = cross_val_score(model_optimized, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

# print ('------------------ MSE for training------------------')
mse_ads = [abs(s) for s in scores]
mse_tr = np.mean(mse_ads)
print ('MSE for training: {:.4f} eV'.format(mse_tr))

# print ('----------------- RMSE -----------------')
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print ('RMSE for training: {:.4f} eV'.format(np.mean(rmse_scores)))
model_optimized.fit(X_train, y_train)

y_tr_pred = model_optimized.predict(X_train)

R_squr_train = r2_score(y_train, y_tr_pred)
print ('R^2 for train {:.4f} eV'.format(R_squr_train))

# 开始做测试
y_te_pred = model_optimized.predict(X_test)

print ('y_te_pred: \n', y_te_pred)

MAE = mean_absolute_error(y_te_pred, y_test)
print ('calcultae average MAE for test: {:.4f} eV'.format(MAE))

MSE = mean_squared_error(y_te_pred, y_test)
print ("calcultae average MSE for test: {:.8f} eV".format(MSE))     # 这个是测试的MAES

RMSE = MSE**.5
print ('calcultae average RMSE for test {:.4f} eV'.format(RMSE))

R_squr = r2_score(y_test, y_te_pred)
print ('R^2 for test {:.4f} eV'.format(R_squr))


# plt.rcParams['figure.figsize'] = (18, 18)
# # plt.rcParams['yticks.fontsize'] = 10
# plot_importance(KNR)


# 画图  
font = {'size': '28'}
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


plt.text(0.6, 0.3, 'KRR Test', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)


plt.legend(loc=0)
# plt.title('KRR')
# plt.savefig('Model_performance_opt_para-0405.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # 500 times training/test

# In[41]:


# R_squr for trainging 
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
    
test1 = pd.DataFrame({'krr-rmse':RMSE_2nd})
test2 = pd.DataFrame({'krr-r2':R2_2nd})
test3 = pd.DataFrame({'krr-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'krr-r2_test':R2_2nd_test})

# writer = pd.ExcelWriter('../model_performance.xlsx', mode='a', engine='xlsxwriter')   # Creating Excel Writer Object from Pandas  
writer = pd.ExcelWriter('../model_performance2.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='krr1',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='krr2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='krr3',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='krr4',startrow=0, startcol=0)
writer.close()  




