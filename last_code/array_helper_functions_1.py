
# coding: utf-8

# In[5]:


#helper functions
import numpy as np

def ind2sub(array_shape, ind):
    rows = ind // array_shape[1]
    cols = ind%array_shape[1] # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def ind2sub_all(array_shape):
    m = array_shape[0]
    n = array_shape[1]
    sub_list = [[] for _ in range(m*n)]
    #Let's get the index of all (i, j) pairs:
    for k in range(m*n):
            sub_list[k] = ind2sub(array_shape, k)
    return sub_list
#print(ind2sub_all(array_shape))
###############################################################
###############################################################
def sub2ind(array_shape, rows, cols):
    return int(rows*array_shape[1] + cols)

def sub2ind_all(array_shape):
    m = array_shape[0]
    n = array_shape[1]
    index_mat = np.zeros((m,n))
    #Let's get the index of all (i, j) pairs:
    for i in range(m):
        for j in range(n):
            index_mat[i,j] = sub2ind(array_shape, i, j) 
    return np.uint32(index_mat)
#array_shape = [5, 6]
#print(sub2ind_all(array_shape))
#print(sub2ind_all(array_shape)[0].dtype)
#################################################################
#################################################################
def min_array2d(arr):
    arr_shape = arr.shape
    min_val   = np.amin(arr)
    min_ind   = (np.abs(arr - min_val)).argmin()
    min_sub   = ind2sub(arr_shape, min_ind)
    return (min_val, min_ind, min_sub)

def max_array2d(arr):
    arr_shape = arr.shape
    max_val   = np.amax(arr)
    max_ind   = (np.abs(arr - max_val)).argmin()
    max_sub   = ind2sub(arr_shape, max_ind)
    return (max_val, max_ind, max_sub)

