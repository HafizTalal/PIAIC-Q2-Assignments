#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[1]:


import numpy as np
np.arange(0,10).reshape(2,5)


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[2]:


import numpy
a = np.arange(10).reshape(2,5)
b = np.ones(10, numpy.int8).reshape(2,5)
np.vstack((a,b))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[3]:


np.hstack((a,b))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[4]:


c = np.arange(10).reshape(5,2)
np.ravel(c)


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[5]:


d = np.arange(15).reshape(5,3)
np.ravel(d)


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[6]:


np.arange(15).reshape((5,3))


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[7]:


e = np.random.rand(5,5)
print(e)
np.square(e)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[8]:


f= np.random.rand(5,6)
print(f)
np.mean(f)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[9]:


np.std(f)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[10]:


np.median(f)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[11]:


f.T


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[13]:


g = np.random.rand(4,4)
print(g)
np.trace(g)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[14]:


np.linalg.det(g)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[15]:


print(np.percentile(g,5))
print(np.percentile(g,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[16]:


np.isin(g,0)

