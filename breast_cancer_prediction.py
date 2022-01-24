#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Prachi7249/BreastCancerPrediction/blob/main/breast_cancer_prediction.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('data.csv')
df.head(10) 


# In[3]:


df.shape


# In[4]:


df.isna().sum()


# In[5]:


df=df.dropna(axis=1)


# In[6]:


df.shape


# In[7]:


df["diagnosis"].value_counts()


# In[8]:


sns.countplot(df["diagnosis"],label="count")


# In[9]:


df.dtypes


# In[10]:


df["diagnosis"].value_counts()


# In[11]:


sns.countplot(df["diagnosis"])


# In[12]:


from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
df.iloc[:,1]=labelencoder_y.fit_transform(df.iloc[:,1].values)


# In[13]:


df.shape


# In[14]:


df.head()


# In[15]:


sns.pairplot(df.iloc[:,1:5], hue="diagnosis")


# In[16]:


df.iloc[:,1:32].corr()


# In[17]:


x=df.iloc[:,2:31].values
y=df.iloc[:,1].values


# In[18]:


# visualize the correlation
plt.figure(figsize=(10,10))
sns.heatmap(df.iloc[:,1:10].corr(),annot=True,fmt=".0%")


# In[19]:


from sklearn.model_selection import train_test_split
x_test,x_train,y_test,y_train=train_test_split(x,y,test_size=0.20,random_state=0)


# In[20]:


from sklearn.preprocessing import StandardScaler
x_train=StandardScaler().fit_transform(x_train)
x_test=StandardScaler().fit_transform(x_test)


# In[21]:


# models/ Algorithms

def models(X_train,Y_train):
        #logistic regression
        from sklearn.linear_model import LogisticRegression
        log=LogisticRegression(random_state=0)
        log.fit(X_train,Y_train)
        
        
        #Decision Tree
        from sklearn.tree import DecisionTreeClassifier
        tree=DecisionTreeClassifier(random_state=0,criterion="entropy")
        tree.fit(X_train,Y_train)
        
        #Random Forest
        from sklearn.ensemble import RandomForestClassifier
        forest=RandomForestClassifier(random_state=0,criterion="entropy",n_estimators=10)
        forest.fit(X_train,Y_train)
        
        print('[0]logistic regression accuracy:',log.score(X_train,Y_train))
        print('[1]Decision tree accuracy:',tree.score(X_train,Y_train))
        print('[2]Random forest accuracy:',forest.score(X_train,Y_train))
        
        return log,tree,forest


# In[22]:


model=models(x_train,y_train)


# In[23]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

for i in range(len(model)):
    print("Model",i)
    print(classification_report(y_test,model[i].predict(x_test)))
    print('Accuracy : ',accuracy_score(y_test,model[i].predict(x_test)))


# In[24]:


# prediction of random-forest
pred=model[0].predict(x_test)
print('Predicted values:')
print(pred)
print('Actual values:')
print(y_test)


# In[25]:


import pickle
pickle.dump(model[0], open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model)


# In[ ]:




