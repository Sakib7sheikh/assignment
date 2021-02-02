#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[3]:


import numpy as np


# 2. Create a null vector of size 10 

# In[4]:


nullvector=np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[5]:


import numpy as np
x= np.arange(10,50)
print(x)


# 4. Find the shape of previous array in question 3

# In[6]:


import numpy as np
x= np.arange(10,50)
x.shape


# 5. Print the type of the previous array in question 3

# In[7]:


import numpy as np
x= np.arange(10,50)
type(x)


# 6. Print the numpy version and the configuration
# 

# In[8]:


import numpy as np
print(np.__version__)
print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[9]:


import numpy as np
x= np.arange(10,50)
x.ndim


# 8. Create a boolean array with all the True values

# In[10]:


import numpy as np
x= np.arange(10,50)
x>9


# 9. Create a two dimensional array
# 
# 
# 

# In[11]:


import numpy as np
y=np.random.randn(2,2)
y


# In[12]:


10. Create a three dimensional array


# In[13]:


import numpy as np
z= np.random.randn(2,2,3)
z


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[14]:


import numpy as np
a=np.arange(1,12)
a=a[::-1]


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[15]:


import numpy as np
b=np.zeros(10)
b[5-1]=1
b


# 13. Create a 3x3 identity matrix

# In[16]:


import numpy as np
c=np.ones((3,3))
c


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[17]:


import numpy as np
arr = np.array([1, 2, 3, 4, 5])
arr=arr.astype('float64')
arr


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

# In[18]:


import numpy as np
arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


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

# In[ ]:





# 17. Extract all odd numbers from arr with values(0-9)

# In[19]:


import numpy as np
d=np.arange(1,10,2)
d


# 18. Replace all odd numbers to -1 from previous array

# In[20]:


import numpy as np
np.array([1,3,5,7,9])=[-1,-1,-1,-1]


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[21]:


import numpy as np
arr = np.arange(10)
arr[[5,6,7,8]]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[22]:


import numpy as np
e=np.zeros((4,4))
e[1:-1,1:-1]=1
e


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

# In[23]:


import numpy as np
arr2d = np.array([[1, 2, 3],

                    [4, 5, 6], 

                    [7, 8, 9]])
arr2d[1][1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[47]:


import numpy as np
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[72]:


import numpy as np
a2d=np.array([1,2,3,4,5,6,7,8,9])
np.reshape(a2d[(-1,5)])
a2d


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[55]:


import numpy as np
b2d=np.array([[0,1,2],[3,4,5],[6,7,8]])
b2d[1][1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[58]:


import numpy as np
b3d=np.array([[0,1,2],[3,4,5],[6,7,8]])
b3d
b3d[(0,0),(1,1)]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[70]:


import numpy as np
x=np.random.randn(10,10)
minimum=np.min(x)
maximum=np.max(x)
print("maximunm=" ,maximum)
print("minimum=" ,minimum)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[75]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
y=np.intersect1d(a,b)
y


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[126]:


import numpy as np
a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[128]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])   
data = np.random.randn(7, 4)
print(names!='Will')
print(data[names!='Will'])


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[127]:


import numpy as np
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])   
data = np.random.randn(7, 4)
b=(names!='Will') & (names!='Joe')
print(b)
print(data[b])


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[89]:


import numpy as np
e2d=np.random.uniform(1,15, size=(5,3))
e2d
#array([[-1.18935238, -0.40600698, -0.80004128],
#       [ 0.77013007, -1.59360643,  0.90270635],
 #      [-0.37481199, -0.08652814,  1.32603341],
  #     [ 2.23347389,  0.11104144,  0.22391165]])


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[91]:


import numpy as np
f3d= np.random.uniform(1,16, size=(2,2,4))
f3d


# 33. Swap axes of the array you created in Question 32

# In[93]:


import numpy as np
z=np.swapaxes(f3d,0,2)
z


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[96]:


import numpy as np
aray=np.array([1,2,3,4,5,6,7,8,9,10])
square=np.square(aray)
square


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[110]:


import numpy as np
g= np.random.uniform(0,12,size=(1,12))
h=np.random.uniform(0,12,size=(1,12))
z=np.maximum(g,h)
z


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[125]:


import numpy as np
array = np.array(['Bob','Joe','Will','Bob','Will','Joe','Joe'])
ar=np.unique(array)
print("unique values :",ar)
b=np.sort(ar)
print(b)


# 37. a = np.array([1,2,3,4,5])
#     b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[122]:


import numpy as np
a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
result= np.setdiff1d(a,b)
result


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

# In[117]:


import numpy as np
sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]]) 

newColumn = np.array([[10,10,10]])

sampleArray[:,1] = newColumn[:,2]
sampleArray


# 
# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[118]:


import numpy as np
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
result= np.dot(x,y)
result


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[121]:


import numpy as np
last = np.random.randn(4,5)
cs= np.cumsum(last)
cs


# In[ ]:




