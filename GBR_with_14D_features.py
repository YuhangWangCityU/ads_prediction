#!/usr/bin/env python
# coding: utf-8

# # Import required modules

# In[2]:


# Import required modulesimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import numpy as np
import random
import os
import torch
from matplotlib.pylab import MultipleLocator
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import KernelRidge
warnings.filterwarnings("ignore")


# In[3]:


# Instantiate regressor algorithms
GBR = GradientBoostingRegressor(random_state=42)
KNR = KNeighborsRegressor()
RFR = RandomForestRegressor(random_state=42)
SR = SVR()
KRR = KernelRidge()
xgbr = XGBRegressor(random_state=42)
light = LGBMRegressor(random_state=42)


# In[4]:


# Read data
file_name='Feature_CO.xlsx'
df = pd.read_excel('../'+file_name, sheet_name= "ML_features")
features = df.iloc[:,:-1]
# delete high correlated features
features3 = features.drop(columns=['$\\mathregular{χ_{NM(0-3)}}$',
                                   '$\\mathregular{m_{NM(0-3)}}$',
                                   '$\\mathregular{r_{NM(3-4)}}$',
                                   '$\\mathregular{m_{NM(3-4)}}$',
                                   '$\\mathregular{χ_{NM(4-5)}}$',
                                   '$\\mathregular{r_{NM(4-5)}}$',
                                   '$\\mathregular{m_{M}}$',
                                    '$\\mathregular{χ_{M}}$',])
#redefine name 
features = features3
target = df.iloc[:,-1]


# In[110]:


# Adding gaussian noise to target values to prevent overfitting
np.random.seed(1)
mu  =  0    # 高斯噪声的均值
sigma = 0.065  # 高斯噪声的方差, 1. 0.065; 2.0.1
# [1, 106], 106是数据的行数
noise = np.random.normal(mu, sigma, [1,108]).tolist()
target_2 = target + noise[0]
# target_2


# # Feature normalization

# In[111]:


# standardize the descriptor 
mean1 = features.mean(axis=0)                              ## 求平均， numpy中的功能 
std1 = features.std(axis=0)                                #  Z标准化：实现中心化和正态分布 
# comment this line if you want unstandardized descriptors  
features = (features - mean1)/ std1                             #  Z标准化：实现中心化和正态分布  
features


# # Model performance for default parameters

# In[1]:


#  data splitting  
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=42) 

# Define the model
model = GBR

kfold = KFold(n_splits=10, shuffle=True, random_state=42)
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
font = {'size': '24'}
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
plt.title('GBR')
# plt.savefig('Model_performance_default_para.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # Hyperparameter tuning 

# In[123]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

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

model_grid = {'loss':['ls', 'lad', 'huber', 'quantile'],
             'n_estimators': [10,50,100],
             'learning_rate':[0.05, 0.1, 0.15],
             'max_depth':[2,3,4],
             'min_samples_split':[2,3,5],
             'min_samples_leaf':[1,2,4]}
model.get_params().keys()
from sklearn.model_selection import RandomizedSearchCV
model_grid_cv = RandomizedSearchCV(estimator=model,
                        param_distributions=model_grid,
                        n_iter=100,
                        cv=10,
                        verbose= True)
model_grid_cv.fit(X_train,y_train)


# In[124]:


model_optimized = model_grid_cv.best_estimator_
print (model)
print (model_optimized)


# #  Model performance after hyperparameter tuning

# In[131]:


# data splitting
X_train, X_test, y_train, y_test = train_test_split(features, target, train_size= 0.8, random_state=42) 


kfold = KFold(n_splits=10, shuffle=False)
scores = cross_val_score(model_optimized, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

# model scores
mse_ads = [abs(s) for s in scores]
mse_tr = np.mean(mse_ads)

# predicted value for y_train
y_tr_pred = model_optimized.predict(X_train)

# print ('----------------- RMSE -----------------')
rmse_scores = np.sqrt(mean_squared_error(y_tr_pred, y_train))
R_squr_train = r2_score(y_train, y_tr_pred)

print ('RMSE for training (v2): {:.4f} eV'.format(rmse_scores))
print ('R^2 for train {:.4f} eV'.format(R_squr_train))


# predicted value for y_test
y_te_pred = model_optimized.predict(X_test)
MAE = mean_absolute_error(y_te_pred, y_test)
MSE = mean_squared_error(y_te_pred, y_test)
RMSE = MSE**.5
R_squr = r2_score(y_test, y_te_pred)
print ('calcultae average RMSE for test {:.4f} eV'.format(RMSE))
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


plt.text(0.5, 0.3, 'GBR Test', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('GBR')
# plt.savefig('Model_performance_opt_para.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # MAE

# In[132]:


error1 = y_te_pred - y_test

def relative_error(y_te_pred, y_test):
    relative_error = np.abs(y_te_pred - y_test)
    return relative_error
relative_error = relative_error(y_te_pred, y_test)

MAE = mean_absolute_error(y_te_pred, y_test)

font={'weight':'normal', 'size': 28}

fig, ax = plt.subplots(figsize=(8.76, 6.5))
n, bins, patches = plt.hist(x=error1, bins='auto', alpha=0.7, rwidth=0.85, density=True)

plt.xlabel('Error (eV)', fontsize=28)
plt.ylabel('Frequency', fontsize=28)

plt.xlim(-1.2, 0.6)
plt.ylim(0, 10)
ax.xaxis.set_major_locator(MultipleLocator(0.3))

plt.yticks(fontsize=24)
plt.xticks(fontsize=24)

plt.grid(axis='y',ls='--',  c='black', alpha=0.3)

plt.text(0.05, 0.9, 'GBR after', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=28)
plt.text(0.05, 0.8, 'MAE = %0.3f eV'% MAE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,  fontsize=28)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 


ax.plot([-0.2, -0.2], [-100, 100], linestyle='--', c='black', alpha=0.3)
ax.plot([0.2, 0.2], [-100, 100], linestyle='--', c='black', alpha=0.3)

# plt.savefig('MAE error frequency.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # 数据分布

# In[135]:


pd.plotting.register_matplotlib_converters()
print("Setup Complete")

fig, ax = plt.subplots(figsize=(8.76, 6.5))

sns.distplot(a=y_train, kde=True, label='Training set')
sns.distplot(a=y_test, kde=True,label='Test set')

plt.legend(loc=0, fontsize=24)
plt.ylim(0,1)

plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 

# plt.title('Distribution of $\mathregular{E_{ad}(CO)}$')
# plt.savefig("Distribution of Ead_CO-2.jpg", dpi=600, bbox_inches='tight')


# # Visualization of decision trees

# In[149]:


from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
import graphviz

# headers
headers = features.columns.values.tolist()

# assume X_train and y_train are your training data
model_optimized
font = {'size':20}
plt.rc("font", **font)

# get the first decision tree in the ensemble
first_tree = model_optimized.estimators_[0][0]

# get the leaf node indices of the tree
leaf_node_indices = first_tree.apply(X_train)

# plot the decision tree
plt.figure(figsize=(15,10))
    
plot_tree(first_tree, feature_names=X_train.columns, filled=True, impurity=True, proportion=True, rounded=True,
           node_ids=True, fontsize=8)  # Change the color to 'green' for all nodes, including leaf nodes

# Get the leaves' index from the tree
leaves = first_tree.apply(X_train) == -1

# Get the matplotlib Patch objects representing the nodes in the tree plot
patches = plt.gca().patches

# Set the face color of the leaf nodes to green
for node_id, patch in enumerate(patches):
    if leaves[node_id]:
        patch.set_facecolor('red')
        
# plt.title('GBR model')
# plt.savefig('tree_0.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # Pearson coeff 

# In[159]:


font = {'size':12}
plt.rc('font', **font)

coeff = features

# data = df.iloc[:, :-1]
# # print (data)
# corr = data.corr()
# print (corr)
corr = coeff.corr()
ax = sns.heatmap(corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=220),annot=True)

ax.set_xticklabels(ax.get_xticklabels(),horizontalalignment='right',rotation=45,size=20)
ax.set_yticklabels(ax.get_xticklabels(),horizontalalignment='right',size=20)

plt.rcParams['figure.figsize'] = (20, 20)
plt.tight_layout()

# fig = ax.get_figure()
# fig.savefig('Pearson.jpeg', dpi=600,)


# # Histogram of feature importance

# In[160]:


headers = features.columns.values.tolist()
headers[0]
len(headers)
# headers
headers


# In[162]:


#  color list
colors = [ 
'#99CCFF',  '#99FF99', 
'#CCCC00', '#CC99FF', '#8FCD13', '#FF99CC', '#4F9186',
'#c86b85', '#878ecd', '#a8e6cf',  '#3f72af', '#f08a5d', 
'#ffde7d',  '#f67280', '#307672', '#fbac91', '#d4a5a5',
'#769fcd',  '#30e3ca', '#FCBAD3', '#aa96da', '#4DBBD5', ]

colors = [ 
  '#878ecd', '#a8e6cf',  '#3f72af', '#f08a5d', 
'#ffde7d',  '#f67280', '#307672', '#fbac91', '#d4a5a5',
'#769fcd',  '#30e3ca', '#FCBAD3', '#aa96da', '#4DBBD5', ]


# In[163]:


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


font={'weight':'normal', 
      'size': 28}
plt.rc('font', **font)


plt.bar(headers_2, importances_2, color=colors)

# ax.set_xlabel("Feature Importance", fontsize = 28)
ax.set_ylabel("Feature Importance", fontsize = 28)
ax.tick_params(axis='y', )
ax.tick_params(axis='x', rotation = 90)

plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 
# plt.title('GBR feature importance')
# plt.title('$\mathregular{χ_{NM}}$(4-5)-χ_M')
plt.grid(axis='y', ls='--', alpha=0.5)


ax.set_ylim(0, 0.30)

plt.tight_layout()
# fig.savefig('Feature importance_all-heng.jpeg', dpi=600,)


# # Pie chart of feature importance

# In[164]:


# 特征重要性饼状图

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
font={'weight':'normal', 'size': 12}

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
                                  autopct='%2.f%%', )

ax.legend(labels_to_plot, bbox_to_anchor=(1, 0, 0, 0.9), fontsize=24, ncol=1)

# plt.setp(ax.texts, size=12)
# plt.setp([t for t in ax.texts if '%' in t.get_text()], ha='center', va='center')

# plt.title('GBR feature importance', fontsize=28)
# fig.savefig('Feature importance.jpg', dpi=600, bbox_inches='tight')
plt.show()


# # standard SHAP values

# In[152]:


import shap
# 假设 model_optimized 是您训练好的GBR模型
explainer = shap.Explainer(model_optimized)
shap_values = explainer.shap_values(features)
# shap_values.values


# In[153]:


np.shape(shap_values)


# In[154]:


shap_values[0:10]


# In[158]:


# 修改图片格式
# fig, ax = plt.subplots(figsize=(8.76, 6.5))  

font={'size': 24}
plt.rc('font', **font)

plt.figure(figsize=(8.75,6.5))
shap.summary_plot(shap_values, features,  show=False, cmap='cool', )

plt.tick_params(labelsize=24)
plt.xlabel('SHAP value (impact on model output)', fontsize=24)
# plt.savefig('./figures/SHAP_图片2.jpeg', dpi=600, bbox_inches='tight' )
plt.show()


# In[206]:


import matplotlib.pyplot as plt

font={'size': 28}
plt.rc('font', **font)

# fig, ax = plt.subplots(figsize=(8.76, 6.5)) 

plt.figure(figsize=(8.75,6.5))
shap.summary_plot(shap_values, X_test,  show=False, 
                 plot_type="bar",)
# plt.savefig('./SHAP feature importance.jpeg', dpi=600, bbox_inches='tight' )
plt.show()


# # PDP 

# In[208]:


import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# 设置字体大小
font = {'size':'56'}
plt.rc ('font', **font)

clf = model_optimized
feature_names = X_test.columns
print ('feature_names', feature_names)
# for i in 
for i in range(13):
    print (i)
    number=i
    features, feature_names = [(number,)], [f"Features #{i}" for i in range(X_test.shape[1])]
    features
    fig, ax = plt.subplots(figsize=(7.5, 6.5)) 
    PartialDependenceDisplay.from_estimator(clf, X_test, features,  ax=ax )
    plt.ylim(-0.5, 0.5)
    print ('features', features)
    
    ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);   ####设置上部坐标轴的粗细
    plt.tick_params(labelsize=48)
    
    # plt.xlabel('SHAP value', fontsize=28)

    plt.ylabel("")
#     plt.savefig(f'./figures/PDP-{number}.jpeg', dpi=600, bbox_inches='tight' )
    plt.show()


# In[166]:
X_test.iloc[sample_index]
# In[167]:
feature_names=X_test.columns
feature_names
# In[178]:
feature_names = X_test.columns
print ('feature_names', feature_names)


# In[179]:
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

# Set the font size
# Create subplots with 1 row and 2 columns
fig, ax = plt.subplots(figsize=(8.76, 6.5)) 
font = {'size': '28'}
plt.rc('font', **font)


clf = model_optimized
feature_names = X_test.columns

number=(13, 11)

# Second subplot
# Select the features for the second subplot
selected_features_2 = [ number ]  # Replace this with the indices of the features for the second subplot
# Create the PartialDependenceDisplay object for the second subplot
display_2 = PartialDependenceDisplay.from_estimator(clf, X_test, selected_features_2, ax=ax)

plt.xticks(fontsize=24)
# plt.savefig(f'./figures/PDP-2D-{number}.jpeg', dpi=600, bbox_inches='tight' )
plt.show()


# In[112]:
# # Prediction of OH without model retraining
# Read data
df3 = pd.read_excel('../Feature_OH.xlsx', sheet_name= "ML_features")
target_OH = df3.iloc[:,-1]
features
model_optimized.fit(features, target_OH)


# In[113]:


X_train, X_test, y_train, y_test = train_test_split(features, target_OH, train_size= 0.8, random_state=42) 
kfold = KFold(n_splits=10, shuffle=False)
scores = cross_val_score(model_optimized, X_train, y_train, scoring='neg_mean_squared_error', cv=kfold)

# print ('------------------ MSE for training------------------')
mse_ads = [abs(s) for s in scores]
mse_tr = np.mean(mse_ads)
print ('MSE for training: {:.4f} eV'.format(mse_tr))

# print ('----------------- RMSE -----------------')
rmse_scores = [np.sqrt(abs(s)) for s in scores]
print ('RMSE for training: {:.4f} eV'.format(np.mean(rmse_scores)))


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

plt.xlim(-3, 2)
plt.ylim(-3, 2)

ax.set_xlabel('DFT (eV)')
ax.set_ylabel('Prediction (eV)')

xpoints = ypoints = plt.xlim()
plt.plot(xpoints, ypoints, linestyle='--', color='k', lw=1, 
        scalex=False, scaley=False)


y_major_locator=MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
x_major_locator=MultipleLocator(1)
ax.xaxis.set_major_locator(x_major_locator)
plt.tick_params(labelsize=24)

ax.spines['bottom'].set_linewidth(2);####设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(2); ####设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(2); 

plt.grid(axis='x', ls='--', alpha=0.3)
plt.grid(axis='y', ls='--', alpha=0.3)


plt.text(0.5, 0.3, 'GBR Test for OH', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('GBR')
# plt.savefig('./performance/OH.jpg', dpi=600, bbox_inches='tight')
plt.show()


# In[114]:
print ('---500 times running ---')
R2_2nd = []
RMSE_2nd = []
R2_2nd_test = []
RMSE_2nd_test = []
for i in range(0,500):
    print (i)
    # 修改该Target
    X_train, X_test, y_train, y_test = train_test_split(features, target_OH, train_size= 0.8) 

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
    
# mean_R2_2nd = np.mean(R2_2nd)
# mean_R2_2nd

test1 = pd.DataFrame({'gbr-rmse':RMSE_2nd})
test2 = pd.DataFrame({'gbr-r2':R2_2nd})
test3 = pd.DataFrame({'gbr-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'gbr-r2_test':R2_2nd_test})

# save data to excel
writer = pd.ExcelWriter('./performance/OH_performance2.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='rmse',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='r2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='rmse_test',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='r2_test',startrow=0, startcol=0)
writer.close()   
print ('---end of 500 times running ---')


# # Prediction of NO without model retraining

# In[108]:


# Read data
df2 = pd.read_excel('../Feature_NO.xlsx', sheet_name= "ML_features")
target_NO = df2.iloc[:,-1]
features
model_optimized.fit(features, target_NO)


# In[109]:


X_train, X_test, y_train, y_test = train_test_split(features, target_NO, train_size= 0.8, random_state=42) 

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

# 画图  
font = {'size': '28'}
plt.rc('font', **font)

fig, ax = plt.subplots(figsize=(8.67, 6.5))
ax.scatter(y_train, y_tr_pred, alpha=0.5, c='#4DBBD5', marker='o', label='Training', s=250)
ax.scatter(y_test, y_te_pred, alpha=0.5, c='#f67280', marker='o', label='Test', s=250)

plt.xlim(-2, 0.5)
plt.ylim(-2, 0.5)

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


plt.text(0.5, 0.3, 'GBR Test for NO', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('GBR')
# plt.savefig('./performance/NO.jpg', dpi=600, bbox_inches='tight')
plt.show()


# In[110]:


print ('---500 times running ---')
R2_2nd = []
RMSE_2nd = []
R2_2nd_test = []
RMSE_2nd_test = []
for i in range(0,500):
    print (i)
    
    # 修改该Target
    X_train, X_test, y_train, y_test = train_test_split(features, target_NO, train_size= 0.8) 

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
    
# mean_R2_2nd = np.mean(R2_2nd)
# mean_R2_2nd

test1 = pd.DataFrame({'gbr-rmse':RMSE_2nd})
test2 = pd.DataFrame({'gbr-r2':R2_2nd})
test3 = pd.DataFrame({'gbr-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'gbr-r2_test':R2_2nd_test})

# save data to excel
writer = pd.ExcelWriter('./performance/NO_performance.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='rmse',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='r2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='rmse_test',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='r2_test',startrow=0, startcol=0)
writer.close()  
print ('---end of 500 times running ---')


# # Prediction of N2 without model retraining

# In[116]:


# Read data
df4 = pd.read_excel('../Feature_N2.xlsx', sheet_name= "ML_features")
target_N2 = df4.iloc[:,-1]
features
model_optimized.fit(features, target_N2)


# In[117]:


X_train, X_test, y_train, y_test = train_test_split(features, target_N2, train_size= 0.8, random_state=42) 
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

plt.xlim(-2, 0.5)
plt.ylim(-2, 0.5)

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


plt.text(0.5, 0.3, 'GBR Test for N$_2$', horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.2, '$RMSE = %0.3f$ eV'% RMSE, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes, fontsize=24)
plt.text(0.5, 0.1, '$R^2 = %0.3f$'% R_squr, horizontalalignment='left',
     verticalalignment='center', transform=ax.transAxes,fontsize=24)

plt.legend(loc=0)
# plt.title('GBR')
plt.savefig('./performance/N2.jpg', dpi=600, bbox_inches='tight')
plt.show()


# In[118]:


print ('---500 times running ---')
R2_2nd = []
RMSE_2nd = []
R2_2nd_test = []
RMSE_2nd_test = []
for i in range(0,500):
    print (i)
    
    # 修改该Target
    X_train, X_test, y_train, y_test = train_test_split(features, target_N2, train_size= 0.8) 

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
    
# mean_R2_2nd = np.mean(R2_2nd)
# mean_R2_2nd

test1 = pd.DataFrame({'gbr-rmse':RMSE_2nd})
test2 = pd.DataFrame({'gbr-r2':R2_2nd})
test3 = pd.DataFrame({'gbr-rmse_test':RMSE_2nd_test})
test4 = pd.DataFrame({'gbr-r2_test':R2_2nd_test})

# save data to excel
writer = pd.ExcelWriter('./performance/N2_performance.xlsx', mode='a', engine='openpyxl')
workbook=writer.book
test1.to_excel(writer,sheet_name='rmse',startrow=0 , startcol=0)   
test2.to_excel(writer,sheet_name='r2',startrow=0, startcol=0)
test3.to_excel(writer,sheet_name='rmse_test',startrow=0, startcol=0)
test4.to_excel(writer,sheet_name='r2_test',startrow=0, startcol=0)
writer.close()   
print ('---end of 500 times running ---')


# In[119]:





# In[ ]:




