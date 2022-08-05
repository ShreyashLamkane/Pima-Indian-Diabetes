#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier


# In[2]:


dataframe=pandas.read_csv("E:/Placement courses/ML Projects/pima Indian Diabetics/pima-indians-diabetes.csv")
array=dataframe.values
X=array[:,0:8]
Y=array[:,8]
seed=7
num_trees=30


# In[3]:


kfold =model_selection.KFold(n_splits=10, random_state=seed)
model=AdaBoostClassifier(n_estimators=num_trees)
results=model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[8]:


import sys
get_ipython().system('{sys.executable} -m pip install xgboost')
from sklearn import svm
from xgboost import XGBClassifier
clf = XGBClassifier()

seed=7
num_trees=30

kfold=model_selection.KFold(n_splits=10)
model = XGBClassifier(n_splits=10)
results=model_selection.cross_val_score(model, X, Y, cv=kfold)
print(results.mean())


# In[ ]:





# In[ ]:




