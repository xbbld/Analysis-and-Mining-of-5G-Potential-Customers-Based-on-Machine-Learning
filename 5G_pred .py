#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,classification_report,roc_curve, auc
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

np.random.seed(1)


# In[2]:


#introduct the data
train1 = pd.read_csv('C:\\Users\\asus\\Downloads\\train.csv')


# In[3]:


#data cleaning 
#clean the Missing value
print(train1.isnull().sum()[train1.isnull().sum()>0]) 
train1.info()


# In[4]:


# Stratified Sampling
each_sample_count = 10000
label_unique = np.unique(train1['target']) 
train = pd.DataFrame(np.zeros((1, len(train1.columns))), columns=train1.columns ,dtype=int)
for label in label_unique:
     sample = pd.DataFrame.sample(train1[train1['target']==label], each_sample_count)
     train = pd.concat([train,sample])


# In[5]:


#normalization
minmax_scaler=preprocessing.MinMaxScaler()
train.iloc[:,20:-1]=minmax_scaler.fit_transform(train.iloc[:,20:-1])


# In[ ]:


#feature engineering
#feature selection
#use RandomForestRegressor to get the value of relative between each feature and target ,
#then delect the feature whose value of relative is low
X = train.iloc[:,0:-1]
Y = train['target']
names = X.columns
rf = RandomForestRegressor(n_estimators=20, max_depth=4)
kfold = KFold(n_splits=5, shuffle=True, random_state=7)
scores = []
train2 =pd.DataFrame(np.zeros((1, len(train.columns))), columns=train.columns ,dtype=int)
for column in X.columns:
    print(column)
    tempx = X[column].values.reshape(-1, 1)
    score = cross_val_score(rf, tempx, Y, scoring="r2",error_score='raise',cv=kfold)
    scores.append((round(np.mean(score), 3), column))
    plt.bar(column,np.mean(score),align='center')
print(sorted(scores, reverse=True))
plt.show()


# In[6]:


#delect the feature whose value of relative is low
train = train.drop('id',axis=1)
train = train.drop('num_16',axis=1)
train = train.drop('num_34',axis=1)
train = train.drop('num_14',axis=1)
train = train.drop('num_17',axis=1)
train = train.drop('num_20',axis=1)
train = train.drop('num_24',axis=1)
train = train.drop('num_28',axis=1)
train = train.drop('num_13',axis=1)
train = train.drop('num_23',axis=1)
train = train.drop('num_35',axis=1)
train = train.drop('cat_2',axis=1)


# In[7]:


#split the data set into 80%train set and 20%test set
x_train,x_test,y_train,y_test = train_test_split(train.iloc[:,:-1],train['target'],test_size=0.2)


# In[12]:


#Train LightGBM model
model_lgb = lgb.LGBMClassifier(learning_rate = 0.12,n_estimators=1010,max_depth=9,num_leaves=140)
model_lgb.fit(x_train,y_train)
score_log = model_lgb.score(x_test,y_test)
y_pred = model_lgb.predict(x_test)
y_score = model_lgb.predict_proba(x_test)[:, 1]
print(roc_auc_score(y_test, y_score))
report_log = classification_report(y_test,y_pred,labels=[0,1],target_names=['不是5G用户','是5G用户'])
print(report_log,
     '\nAUC指标为%.4f' % roc_auc_score(y_test, y_score))


# In[13]:


#Train RandomForestClassifier model
estimator_forest = RandomForestClassifier(n_estimators =412,max_depth=31,min_samples_split=5)
estimator_forest.fit(x_train,y_train)
y_pred = estimator_forest.predict(x_test)
report_forest = classification_report(y_test,y_pred,labels=[0,1],target_names=['不是5G用户','是5G用户'])
auc_forest = roc_auc_score(y_test,estimator_forest.predict_proba(x_test)[:, 1])
print(report_forest,
     '\nAUC指标为%.4f：' % auc_forest)


# In[ ]:


#paint the picture of auc(Area Under Curve)
fpr, tpr, _ = roc_curve(y_test, y_pred)  
roc_auc = auc(fpr, tpr) 
plt.figure()  
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)  
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
plt.xlim([0.0, 1.0])  
plt.ylim([0.0, 1.05])  
plt.xlabel('FPR')  
plt.ylabel('TPR')  
plt.title('ROC')  
plt.legend(loc="lower right")  
plt.show()


# In[14]:


'''
#paint the picture of auc in different parameters
#determine the range of Optimal parameters
cross = []
for i  in np.arange(130,150,2):
    model_lgb = lgb.LGBMClassifier(learning_rate = 0.12,n_estimators=1010,max_depth=9,num_leaves=i)
    model_lgb.fit(x_train,y_train)
    y_pred = model_lgb.predict(x_test)
    cross.append(roc_auc_score(y_test, y_pred))
plt.plot(np.arange(130,150,2),cross)
plt.xlabel('max_depth')
plt.ylabel('auc')
plt.show()
'''


# In[ ]:


'''
#depending on the above-mentioned range of the Optimal parameters,use GridSearchCV to find out the Optimal parameters
param_test1 = {'max_depth':range(3,10,1),'num_leaves':range(1,100,10)
               }
gsearch1 = GridSearchCV(estimator=lgb.LGBMClassifier(learning_rate = 0.12,n_estimators=1060),param_grid=param_test1,
                        scoring='roc_auc')
gsearch1.fit(x_train,y_train)
print(gsearch1.score)
print(gsearch1.best_params_)
'''


# In[ ]:




