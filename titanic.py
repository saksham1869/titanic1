#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


test_df = pd.read_csv("/test.csv")
train_df = pd.read_csv("/train.csv")


# In[5]:


train_df.head()


# In[6]:


train_df.shape


# In[7]:


train_df.isnull().sum()


# In[8]:


train_df = train_df.drop(columns = 'Cabin', axis = 1)


# In[9]:


train_df['Age'].fillna(train_df['Age'].mean(), inplace = True)


# In[10]:


train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)


# In[11]:


train_df.isnull().sum()


# Exploratory Data Analysis

# In[12]:


train_df = train_df.drop(columns = 'Name', axis=1)


# In[13]:


train_df.describe(include='all')


# In[14]:


train_df['Survived'].value_counts()


# Data Visualisation

# In[15]:


sns.set()
sns.countplot('Survived', data=train_df)


# In[16]:


sns.countplot('Sex', data=train_df)


# In[17]:


sns.countplot('Sex', hue='Survived', data=train_df)


# In[18]:


sns.countplot('Pclass', data=train_df)


# In[19]:


sns.countplot('Pclass', hue='Survived', data=train_df)


# In[20]:


train_df.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[21]:


X = train_df.drop(columns = ['PassengerId','Ticket','Survived'],axis=1)
Y = train_df['Survived']


# In[22]:


X.head()


# In[23]:


Y.head()


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# Model Training

# In[25]:


model = LogisticRegression()
model.fit(X_train, Y_train)


# In[26]:


X_train_prediction = model.predict(X_train)


# In[27]:


X_train_prediction


# In[28]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
training_data_accuracy


# In[29]:


X_test_prediction = model.predict(X_test)


# In[30]:


X_test_prediction


# In[31]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
test_data_accuracy


# In[ ]:





# In[ ]:




