#!/usr/bin/env python
# coding: utf-8

# # Import required modules

# In[1]:


import pandas as pd
import numpy as np 
import time
import matplotlib.pyplot as plt
import warnings
import os 
import string 
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
from sklearn.model_selection import RandomizedSearchCV

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


# In[4]:


# Adding gaussian noise to target values to prevent overfitting
np.random.seed(1)
mu  =  0    # 高斯噪声的均值
sigma = 0.065  # 高斯噪声的方差, 1. 0.065; 2.0.1

# [1, 106], 106是数据的行数
noise = np.random.normal(mu, sigma, [1,108]).tolist()
target_2 = target + noise[0]


# # Feature normalization

# In[5]:


# Feature normalization# standardize the descriptor 
mean1 = features.mean(axis=0)                              ## 求平均， numpy中的功能 
std1 = features.std(axis=0)                                #  Z标准化：实现中心化和正态分布 
# comment this line if you want unstandardized descriptors  
features = (features - mean1)/ std1                             #  Z标准化：实现中心化和正态分布  


# # Model performance for default parameters

# In[11]:


#  data splitting  
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=42) 

# define the model 
model = RFR
kfold = KFold(n_splits=10, shuffle=False)
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

plt.text(0.5, 0.3, 'RFR Test', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('RFR')
# plt.savefig('Model_performance_default_para_new-0405.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # Hyperparameter tuning 

# In[14]:


# Hyperparameter tuning # Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
model_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': [2,3,4],
               'min_samples_split': [2,3,5],
               'min_samples_leaf': [1,2,4,5,6],
               'bootstrap': bootstrap}
print (model)
model.get_params().keys()


# In[15]:



model_grid_cv = RandomizedSearchCV(estimator=model,
                        param_distributions=model_grid,
                        n_iter=100,
                        cv=10,
                        verbose= True)
model_grid_cv.fit(X_train,y_train)
model_optimized = model_grid_cv.best_estimator_
print (model_optimized)


# #  Model performance after hyperparameter tuning

# In[22]:


# 重新定义数据的分割情况, 主要是调整 random_state
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=47) #  Model performance after hyperparameter tuning
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
RFR.fit(X_train, y_train)

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
ax.scatter(y_train, y_tr_pred, alpha=0.5, c='b', marker='o', label='Training')
ax.scatter(y_test, y_te_pred, alpha=0.5, c='r', marker='o', label='Test')

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

plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
plt.title('RFR')
# plt.savefig('Model_performance_opt_para-0405.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # Histogram of feature importance

# In[23]:


headers = features.columns.values.tolist()
headers[0]
len(headers)
# headers


# In[24]:


# importances = model_optimized.feature_importances_
# headers = headers 
# print (len(headers))

# sorted_idx = np.argsort(importances)
# sorted_idx

# importances_2 = importances[sorted_idx]
# print (importances_2)

# headers_2 = np.array(headers)[sorted_idx]
# # headers = headers[len(headers)]
# print (headers_2)

# fig, ax = plt.subplots( figsize=(6.5, 8.75))
# ax.barh(headers_2,
#         importances_2,
#         height=0.7)

# ax.set_xlabel("Feature Importance", fontsize = 28)
# ax.tick_params(axis='y', size = 2)

# plt.tight_layout()
# # fig.savefig('Feature importance_all.jpeg', dpi=600,)


# In[25]:


# color list
colors = [ 
'#99CCFF',  '#99FF99', 
'#CCCC00', '#CC99FF', '#8FCD13', '#FF99CC', '#4F9186',
'#c86b85', '#878ecd', '#a8e6cf',  '#3f72af', '#f08a5d', 
'#ffde7d',  '#f67280', '#307672', '#fbac91', '#d4a5a5',
'#769fcd',  '#30e3ca', '#FCBAD3', '#aa96da', '#4DBBD5', ]


# In[26]:


importances = model_optimized.feature_importances_
headers = headers 
print (len(headers))

sorted_idx = np.argsort(importances)
sorted_idx

importances_2 = importances[sorted_idx]
print (importances_2)

headers_2 = np.array(headers)[sorted_idx]
# headers = headers[len(headers)]
print (headers_2)

fig, ax = plt.subplots( figsize=(15, 8))
# ax.barh(importances_2,
#         headers_2,
#         height=0.7)

plt.bar(headers_2, importances_2, )

# ax.set_xlabel("Feature Importance", fontsize = 28)
ax.set_ylabel("Feature Importance", fontsize = 28)
ax.tick_params(axis='y', size = 2)
ax.tick_params(axis='x', rotation = 90)


plt.title('RFR feature importance')
plt.grid(axis='y', ls='--', alpha=0.5)

plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 

ax.set_ylim(0, 0.20)

plt.tight_layout()
# fig.savefig('Feature importance_all-heng.jpeg', dpi=600,)


# # Pie chart of feature importance

# In[27]:


# Pie chart of feature importance
model_optimized = model

# 默认是从小到大的排序
sorted_idx = np.argsort(importances)
sorted_idx

# sort 排序得到的 index， 而不是具体的数
importances_2 = importances[sorted_idx]
importances_2

# pie
feature_numbers = 13
idx_third_tem = model_optimized.feature_importances_.argsort()   # 对特征重要性排序  
idx_third = idx_third_tem[-feature_numbers]          # 测试一下， 最后一个是多少

# 得到从大到小的index
feature_idx = (-model_optimized.feature_importances_).argsort()[:feature_numbers] 

feature_idx

# 对importance 加负号，然后排序，得到的index就是 从大到小的
(-model_optimized.feature_importances_) 

(-model_optimized.feature_importances_).argsort()

# 得到特征的名字
name_features = np.array(headers)[feature_idx]  
name_features

(-model_optimized.feature_importances_).argsort()[:feature_numbers] 

# 特征重要性的具体的数据
selected_features = model_optimized.feature_importances_[feature_idx]
selected_features



# 只画前 n 个重要的特征，而将其他的合并为others
font={'weight':'normal', 
      'size': 12}

cmap = plt.get_cmap("tab20")

colors = cmap(np.arange(len(headers)))

labels= name_features  ## np.array(headers)[tree_importance_sorted_idx[3:]]
sizes= selected_features

#  不同块之间的分割
myexplode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05,]

fig, ax = plt.subplots(figsize=(15, 10))

top_sizes = sizes[:9]
top_labels = labels[:9]

other_label = 'Others'
other_size = sum(sizes[9:])


labels_to_plot = np.append(top_labels, [other_label])
labels_to_plot

sizes_to_plot = np.append(top_sizes, [other_size])
sizes_to_plot

wedges,texts, autotexts = plt.pie(sizes_to_plot, 
#                                    colors=colors[:11], 
                                  colors=['#4DBBD5', '#aa96da','#FCBAD3','#30e3ca','#769fcd', 
                                           '#d4a5a5','#fbac91','#307672',  '#f67280',  '#ffde7d', ],
                                   explode = myexplode[:10], 
                                   startangle=90, 
                                   textprops=font,
                                  autopct='%2.f%%',)


ax.legend(labels_to_plot, bbox_to_anchor=(1, 0, 0, 0.9), fontsize=24, ncol=1)

# plt.title('GBR feature importance', fontsize=28)
# fig.savefig('Feature importance.jpg', dpi=600, bbox_inches='tight')
plt.show()


# In[28]:


# 只画前 n 个重要的特征，而将其他的合并为others
font={'weight':'normal', 
      'size': 32}

cmap = plt.get_cmap("tab20")

colors = cmap(np.arange(len(headers)))

labels= name_features  ## np.array(headers)[tree_importance_sorted_idx[3:]]
sizes= selected_features

#  不同块之间的分割
myexplode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
             0.05, 0.05,]

fig, ax = plt.subplots(figsize=(15, 10))

top_sizes = sizes[:3]
top_labels = labels[:3]

other_label = 'Others'
other_size = sum(sizes[3:])

labels_to_plot = np.append(top_labels, [other_label])
labels_to_plot

sizes_to_plot = np.append(top_sizes, [other_size])
sizes_to_plot

wedges,texts, autotexts = plt.pie(sizes_to_plot, 
#                                    colors=colors[:11], 
                                  colors=['#4DBBD5', '#aa96da','#FCBAD3', '#d4a5a5',],
                                   explode = myexplode[:4], 
                                   startangle=90, 
                                   textprops=font,
                                  autopct='%2.f%%',)

ax.legend(labels_to_plot, bbox_to_anchor=(1, 0, 0, 0.9), fontsize=42, ncol=1)

# plt.title('GBR feature importance', fontsize=28)
# fig.savefig('Feature importance_top_3.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # 500 times training/test

# In[32]:


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

test1 = pd.DataFrame({'rfr-rmse':RMSE_2nd})
test2 = pd.DataFrame({'rfr-r2':R2_2nd})
test3 = pd.DataFrame({'rfr-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'rfr-r2_test':R2_2nd_test})

# Creating Excel Writer Object from Pandas  
writer = pd.ExcelWriter('../model_performance2.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='rfr1',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='rfr2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='rfr3',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='rfr4',startrow=0, startcol=0)
writer.close()    


# In[33]:




