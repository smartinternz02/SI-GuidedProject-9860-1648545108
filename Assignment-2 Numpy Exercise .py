#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # NumPy Exercises 
# 
# Now that we've learned about NumPy let's test your knowledge. We'll start off with a few simple tasks, and then you'll be asked some more complicated questions.

# #### Import NumPy as np

# In[1]:


import numpy as np


# #### Create an array of 10 zeros 

# In[2]:


np.zeros(10)


# #### Create an array of 10 ones

# In[3]:


np.ones(10)


# #### Create an array of 10 fives

# In[4]:


np.ones(10) * 5


# #### Create an array of the integers from 10 to 50

# In[5]:


np.arange(10,51)


# #### Create an array of all the even integers from 10 to 50

# In[6]:


np.arange(10,51,2)


# #### Create a 3x3 matrix with values ranging from 0 to 8

# In[7]:


np.arange(9).reshape(3,3)


# #### Create a 3x3 identity matrix

# In[8]:


np.eye(3)


# #### Use NumPy to generate a random number between 0 and 1

# In[9]:


np.random.rand(1)


# #### Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution

# In[10]:


np.random.randn(25)


# #### Create the following matrix:

# In[11]:


np.arange(1,101).reshape(10,10) / 100


# #### Create an array of 20 linearly spaced points between 0 and 1:

# In[12]:


np.linspace(0,1,20)


# ## Numpy Indexing and Selection
# 
# Now you will be given a few matrices, and be asked to replicate the resulting matrix outputs:

# In[13]:


mat = np.arange(1,26).reshape(5,5)
mat


# In[14]:


mat[2:,1:]


# In[15]:


mat[3,4]


# In[16]:


mat[:3,1:2]


# In[18]:


mat[3:5,:]


# ### Now do the following

# #### Get the sum of all the values in mat

# In[19]:


mat.sum()


# #### Get the standard deviation of the values in mat

# In[20]:


mat.std()


# #### Get the sum of all the columns in mat

# In[21]:


mat.sum(axis=0)


# # Great Job!
