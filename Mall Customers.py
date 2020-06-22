#!/usr/bin/env python
# coding: utf-8

# In[83]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
import numpy as np
from sklearn.cluster import KMeans
get_ipython().run_line_magic('matplotlib', 'inline')


# In[84]:


df = pd.read_csv('Mall_Customers.csv')
df


# In[85]:


ax = sns.catplot(x='Gender',kind='count',data=df,orient="h")
ax.fig.autofmt_xdate()


# In[110]:


sns.set(style="darkgrid")
sns.catplot(x = "Age", y = "Annual Income (k$)", kind = 'bar', hue= 'Gender', data = df, height=5, aspect=3)
plt.title('Relation btween Age and Annual Income')


# In[111]:


sns.set(style="darkgrid")
sns.relplot(x = "Age", y = "Spending Score (1-100)", kind = 'line', hue= 'Gender', data = df, height=5, aspect=3)
plt.title('Relation btween Age and Spending Score')


# In[88]:


y = df.iloc[:, -1]
y


# In[89]:


df.Gender[df.Gender == "Male"] = 0
df.Gender[df.Gender == 'Female'] = 1
print(df)


# In[90]:


x = df.iloc[:,0:4]
x


# In[91]:


kmeans = KMeans(5)
kmeans.fit(x)


# In[92]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[93]:


data_with_clusters = df.copy()
# Create a new column and add the identified cluster array in the dataset
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[102]:


plt.scatter(data_with_clusters['Age'],data_with_clusters['Annual Income (k$)'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.title('Clusters of Annual Income with respect to various Age Groups')  
plt.xlabel('Age')  
plt.ylabel('Annual Income(k$)')
plt.show()


# In[101]:


plt.scatter(data_with_clusters['Annual Income (k$)'],data_with_clusters['Spending Score (1-100)'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.title('Clusters of Spending Score with respect to Annual Income')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score')
plt.show()

