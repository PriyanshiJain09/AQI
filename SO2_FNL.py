#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[2]:


data=pd.read_csv(r"D:\My Documents\Desktop\AQI.csv")


# In[3]:


n_row,n_col=data.shape
print("The data has {} rows {} cols.".format(n_row,n_col))


# In[4]:


data.describe()


# In[5]:


data.isna().sum()


# In[6]:


data.dropna(inplace=True)
data.isna().sum()


# In[7]:


data.drop(data[data['pollutant_id']!='SO2'].index,inplace=True)


# In[8]:


data


# In[9]:


df = data.groupby(['city', 'pollutant_id']).agg({'pollutant_min': 'min', 'pollutant_max': 'min'})
df['pollutant_avg'] = (df['pollutant_max'] + df['pollutant_min']) / 2


# In[10]:


df


# In[11]:


df['pollutant_min'].unique()


# In[12]:


df['pollutant_max'].unique()


# In[13]:


df['pollutant_avg'].unique()


# In[14]:


num_col = df.select_dtypes(include=['float64', 'int64']).columns
boxplot = df.boxplot()


# In[15]:


for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outlier_mask = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))
    if outlier_mask.any():
        df = df[~outlier_mask] 
        print("Column '{}' has outliers.".format(col))


# In[16]:


bins = [0, 40, 80, 380, 800, np.inf]
categories = ['Good', 'Satisfactory', 'Moderately Polluted', 'Poor', 'Very Poor']


# In[17]:


df['pollutant_level'] = pd.cut(df['pollutant_avg'], bins=bins, labels=categories, right=False)


# In[18]:


df


# In[19]:


X = df[['pollutant_min', 'pollutant_max', 'pollutant_avg']]
y = df['pollutant_level']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[21]:


classes = np.unique(np.concatenate((y_test, y_pred)))

conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)

conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_df, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[ ]:




