#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


np.zeros((10))


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


a = np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[4]:


a.shape


# 5. Print the type of the previous array in question 3

# In[5]:


a.dtype


# 6. Print the numpy version and the configuration
# 

# In[6]:


print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[7]:


a.ndim


# 8. Create a boolean array with all the True values

# In[8]:


np.array((4,2), bool)


# 9. Create a two dimensional array
# 
# 
# 

# In[9]:


np.random.randn(2,3)


# 10. Create a three dimensional array
# 
# 

# In[10]:


np.random.randn(2,2,4)


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[11]:


a[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[12]:


b = np.zeros(10)
b[4]= 1


# 13. Create a 3x3 identity matrix

# In[13]:


np.identity(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[14]:


arr = np.array([1, 2, 3, 4, 5])
arr.astype(float)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[15]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1 * arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[16]:


arr1>arr2


# 17. Extract all odd numbers from arr with values(0-9)

# In[22]:


c = np.arange(0,10)
c[1:10:2]


# 18. Replace all odd numbers to -1 from previous array

# In[23]:


c[1:10:2] = -1


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[24]:


arr = np.arange(10)
arr[5:9] = 12


# 20. Create a 2d array with 1 on the border and 0 inside

# In[25]:


d = np.ones((4,4))
d[1:3, 1:3] = 0


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[26]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1,1] = 12


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[27]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,0,] = 64


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[28]:


d = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(2,5)
print(d)
d[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[30]:


d = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(2,5)
d[1,1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[31]:


d = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(2,5)
d[:,2:3]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[32]:


e = np.random.randn(10,10)
print(e)
print(np.max(e))
print(np.min(e))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[33]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a,b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[34]:


print(np.where(a == b))
a == b


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[35]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(names)
print(data)
data[names != 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[36]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
print(names)
print(data)
data[(names != 'Will') & (names != 'Joe')]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[37]:


np.arange(1,16, dtype = float).reshape(5,3)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[40]:


f = np.arange(1, 17, dtype = float).reshape(2,2,4)


# 33. Swap axes of the array you created in Question 32

# In[41]:


f.swapaxes(1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[42]:


g = np.random.rand(10)
print(g)
h = np.sqrt(g)
h[h<0.5]= 0


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[44]:


i = np.random.rand(12)
j = np.random.rand(12)
print(i)
print(j)
np.maximum(i,j)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[45]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
(np.unique(names))


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[46]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[48]:


import numpy
sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]])

newColumn = numpy.array([[10,10,10]])
k = np.delete(sampleArray,1,1)
print(k)
np.insert(k,1,newColumn,1)


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[49]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
x.dot(y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[50]:


l = np.random.rand(20)
np.cumsum(l)

