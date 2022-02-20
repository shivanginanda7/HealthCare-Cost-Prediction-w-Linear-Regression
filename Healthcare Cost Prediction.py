#!/usr/bin/env python
# coding: utf-8

# ## HealthCare Cost Prediction w/ Linear Regression

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# ### Data Collection & Analysis

# In[4]:


data = pd.read_csv('C:/Users/HP/insurance.csv')


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.isnull().any()


# In[9]:


data.isnull().sum()


# ### Data Analysis

# In[10]:


data.describe()


# In[12]:


sns.set()
plt.figure(figsize=(7,7))
sns.distplot(data['age'])
plt.title('Age Distribution')
plt.show()


# In[17]:


plt.figure(figsize=(5,5))
sns.countplot(x='sex',data=data)
plt.title('Sex Distribution')
plt.show()


# In[14]:


data['sex'].value_counts()


# In[15]:


plt.figure(figsize=(7,7))
sns.distplot(data['bmi'])
plt.title('BMI Distribution')
plt.show()


# In[ ]:


BMI= kg/m^2
Normal BMI Range: 18.5 to 24.9


# In[18]:


plt.figure(figsize=(5,5))
sns.countplot(x='children',data=data)
plt.title('Children')
plt.show()


# In[20]:


data['children'].value_counts()


# In[21]:


plt.figure(figsize=(5,5))
sns.countplot(x='smoker',data=data)
plt.title('Smokers')
plt.show()


# In[22]:


data['smoker'].value_counts()


# In[23]:


plt.figure(figsize=(5,5))
sns.countplot(x='region',data=data)
plt.title('Region')
plt.show()


# In[24]:


data['region'].value_counts()


# In[25]:


sns.set()
plt.figure(figsize=(7,7))
sns.distplot(data['charges'])
plt.title('Charges Distribution')
plt.show()


# ### Data Pre-Processing

# In[30]:


# encoding 'sex' column
data.replace({'sex':{'male':0, 'female':1}}, inplace=True)

## encoding 'smoker' column
data.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)

## encoding 'region' column
data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}},inplace=True)


# ### Splitting the Features and Target

# In[29]:


X= data.drop(columns='charges',axis=1)
Y= data['charges']


# In[31]:


print(X)


# In[32]:


print(Y)


# ### Spliting data into Training & Testing Data

# In[33]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=2)


# In[34]:


print(X.shape, X_train.shape, X_test.shape)


# ### Model Training using Linear Regression

# In[35]:


regressor= LinearRegression()


# In[36]:


regressor.fit(X_train, Y_train)


# ### Model Evaluation

# In[37]:


training_data_prediction= regressor.predict(X_train)


# In[39]:


r2_train = metrics.r2_score(Y_train,training_data_prediction)
print('R Square Value: ',r2_train)


# In[40]:


test_data_prediction= regressor.predict(X_test)


# In[41]:


r2_test = metrics.r2_score(Y_test,test_data_prediction)
print('R Square Value: ',r2_test)


# ### Building a Predicitve Sysytem

# In[50]:


input_data= (31,1,25.74,0,1,0)

#tuple to array
Z= np.array(input_data)

#reshape array
R= Z.reshape(1,-1)

prediction = regressor.predict(R)
print('Insurance Cost: ',prediction)


# In[ ]:




