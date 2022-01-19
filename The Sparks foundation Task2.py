#!/usr/bin/env python
# coding: utf-8

# Task2: Unsupervised Machine Learning- K-Means Clustering

# Problem Statement: From the given "Iris" dataset,predict optimum number of clusters and represent it visually.
# 

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Iris (1).csv")


# Importing the dataset

# In[3]:


df.head()


# In[4]:


##Drop "Species" and "id"column as it has no role in clustering
df.drop(["Species","Id"],axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# Data Preprocessing
# 

# In[8]:


df.isnull().sum()         ## there is no null values


# optimum value of K (elbow method)

# In[9]:


from sklearn.cluster import KMeans


# In[10]:


list=[]
for i in range(1,6):
    model=KMeans(n_clusters=i,n_init=10,max_iter=500)
    model.fit(df)
    list.append(model.inertia_)


# In[11]:


#WCSS values are stored in list
list


# In[21]:


# Finding the optimum number of clusters for k-means classification

x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# In[13]:


#the first lowest decrese in wcss values is k=3,no. of clusters=3


# K MEANS CLUSTERING MODEL

# In[14]:


k_model=KMeans(n_clusters=3,n_init=10,max_iter=500)
k_model.fit(df)


# In[15]:


#cluster centroids
centroids=k_model.cluster_centers_
centroids


# In[16]:


prediction=k_model.predict(df)
df['clusters']=prediction


# In[17]:


df.head()


# In[18]:


#scatter plot representing all the clusters


# In[22]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[23]:


# Visualising the clusters - On the first two columns
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')

plt.legend()


# In[ ]:




