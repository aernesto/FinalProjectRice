
# coding: utf-8

# In[4]:


import numpy as np
from array_helper_functions import ind2sub, ind2sub_all, sub2ind, sub2ind_all
#toy example of fixImages_stackocclusions
#a = np.arange(16).reshape(4, 2, 2)
#b = a + 2
#print("a")
#print(a)
#print(2*'\n')
#print("b")
#print(b)
#print(2*'\n')

#print("stack_axis = 0")
#print(np.stack((a,b), axis=0))
#print(2*'\n')
#print("stack_axis = 1")
#print(np.stack((a,b), axis=1))
#print(2*'\n')

#This is exactly what we need:
#print("stack_axis = 2")
#print(np.stack((a,b), axis=2))


# In[15]:


#fixImages_stackocclusions function
#Stack each image along with all of its occulusions!
#This function is more useful for display purposes!
#n  = 3
#n1 = n2 = 4
#imgs    = np.arange(n*n1*n2).reshape(n,n1,n2)

#print(imgs)
#print(2*'\n')
#print(imgs[0]) #first image 
#print(imgs[1]) #second image
#print(imgs[2]) #third image

#strides    = [1, 1]
#patch_size = [2, 2]

def get_occ_dims(strides, patch_size):
    iter_columns   = range(0, 28-patch_size[0]+strides[0], strides[0]) #iterator along columns object 
    iter_rows      = range(0, 28-patch_size[1]+strides[1], strides[1]) #iterator along rows object
    M2             = len(iter_columns) #num of occlusions row_wise
    M1             = len(iter_rows) #num of occlusions column_wise
    return (M1, M2)

def get_patch_pins(arr_shape, strides, patch_size):
    strides1 = strides[0] #stride along columns
    strides2 = strides[1] #stride along rows
    
    w  = patch_size[0]  #width of the patch
    h  = patch_size[1]  #height of the patch
    
    n1 = arr_shape[0] # the first dimension of image
    n2 = arr_shape[1] # the second dimension of image
    
    iter_columns   = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows      = range(0, n2-h+strides2, strides2) #iterator along rows object

    #total number of possible occlusions in each dimension 
    num_columns       = len(iter_columns) #total number of iterations along columns
    num_rows          = len(iter_rows)  #total number of iterations along rows
    
    index_mat      = sub2ind_all([num_rows, num_columns])  
    patch_pins_mat = np.zeros((num_rows*num_columns, 2, 2))
    #we want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for j in iter_columns:
        for i in iter_rows:
            
            if j == iter_columns[-1]:
                patch_pins_mat[index_mat[i,j]] = np.array([[i, i+h],[j, n2]])
                
            elif i == iter_rows[-1]:
                patch_pins_mat[index_mat[i,j]] = np.array([[i, n1], [j, j+w]])
            
            else:
                patch_pins_mat[index_mat[i,j]] = np.array([[i, i+h], [j, j+w]])
    return np.uint32(patch_pins_mat)
#arr_shape = (28, 28)
#strides = (1, 1)
#patch_size = (5,5)
#print(get_patch_pins(arr_shape, strides, patch_size))
#print(get_patch_pins(arr_shape, strides, patch_size).dtype)
    
def fixImages_stackOcclusions(imgs, strides, patch_size):
    strides1 = strides[0] #stride along columns
    strides2 = strides[1] #stride along rows
    
    w  = patch_size[0]  #width of the patch
    h  = patch_size[1]  #height of the patch
    
    n  = imgs.shape[0] # total number of images that belong to imgs
    n1 = imgs.shape[1] # the first dimension of each image
    n2 = imgs.shape[2] # the second dimension of each image
    
    
    GivenImages_stackOcclusions = [] #initializing a list for storing occluded images 
    
    iter_columns = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows    = range(0, n2-h+strides2, strides2) #iterator along rows object

    #total number of possible occlusions in each dimension 
    num_columns     = len(iter_columns) #total number of iterations along columns
    num_rows        = len(iter_rows)  #total number of iterations along rows


    #we want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for j in iter_columns:
        for i in iter_rows:

            occluded_imgs = imgs.copy()
            if j == iter_columns[-1]:
                occluded_imgs[:,i:i+h, j:n2] = 0
                #print(range(i, n2), range(j, j+h))
                #print(occluded_imgs)
                #print(2*'\n')
                GivenImages_stackOcclusions.append(occluded_imgs)
                
            elif i == iter_rows[-1]:
                occluded_imgs[:,i:n1, j:j+w] = 0
                #print(range(i, i+w), range(j,n1))
                #print(occluded_imgs)
                #print(2*'\n')
                GivenImages_stackOcclusions.append(occluded_imgs)
            
            else:
                occluded_imgs[:, i:i+h, j:j+w] = 0
                #print(range(i, i+w), range(j,j+w))
                #print(occluded_imgs)
                #print(2*'\n')
                GivenImages_stackOcclusions.append(occluded_imgs)
    #print(len(GivenImages_stackOcclusions))
    #print(num_rows*num_columns)
    #print(GivenImages_stackOcclusions)
    
    #If were to stack each image along with all of its occulusions then this would have been what we wanted!
    #print('axis=1')
    #print(np.stack(GivenImages_stackOcclusions, axis=1))
    stack_occlusions_array = np.stack(GivenImages_stackOcclusions, axis=1)
    return stack_occlusions_array

#stack_occlusions_array = fixImages_stackOcclusions(imgs, strides, patch_size)
#print("stack_occlusions_array")
#print(stack_occlusions_array)


# In[13]:


#fixOcclusions_stackImages
# but given an occlusion located in a particular area in the image, for all images in our digitClass
#we want to list that particular occlusion for all the images in our digitClass

#n  = 3
#n1 = n2 = 4
#imgs    = np.arange(n*n1*n2).reshape(n,n1,n2)

#print("all images together in one array:")
#print(imgs)
#print(2*'\n')

#print("first image:")
#print(imgs[0]) #first image 
#print('\n')

#print("second image")
#print(imgs[1]) #second image
#print('\n')

#print("third image")
#print(imgs[2]) #third image
#print('\n')

#strides    = [1, 1]
#patch_size = [2, 2]

#for each fixed occlusion, we are stacking the occluded images with that particular occlusion
#in a single entry of a list
def fixOcclusions_stackImages(imgs, strides, patch_size):
    strides1 = strides[0] #stride along columns
    strides2 = strides[1] #stride along rows
    
    w  = patch_size[0]  #width of the patch
    h  = patch_size[1]  #height of the patch
    
    #
    n  = imgs.shape[0] # total number of images that belong to imgs
    n1 = imgs.shape[1] # the first dimension of each image
    n2 = imgs.shape[2] # the second dimension of each image
    
    iter_columns = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows    = range(0, n2-h+strides2, strides2) #iterator along rows object
    #total number of possible occlusions in each dimension 
    num_columns  = len(iter_columns) #total number of iterations along columns
    num_rows     = len(iter_rows)  #total number of iterations along rows
    M            = num_rows * num_columns
    
    index_mat    = sub2ind_all([num_rows, num_columns])
    #initializing an array for storing occluded images
    givenOcclusions_stackImages = np.zeros((M, n, n1, n2)) 
    #We want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for j in iter_columns:
        for i in iter_rows:

            occluded_imgs = imgs.copy()
            if j == iter_columns[-1]:
                occluded_imgs[:, i:i+h, j:n2] = 0
                #print(range(i, n2), range(j, j+h))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages[index_mat[i,j]] = occluded_imgs
                
            elif i == iter_rows[-1]:
                occluded_imgs[:, i:n1, j:j+w] = 0
                #print(range(i, i+w), range(j,n1))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages[index_mat[i,j]] = occluded_imgs
            else:
                occluded_imgs[:, i:i+h, j:j+w] = 0
                #print(range(i, i+w), range(j,j+w))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages[index_mat[i,j]] = occluded_imgs
                
    return givenOcclusions_stackImages

#givenOcclusions_stackImages = fixOcclusions_stackImages(imgs, strides, patch_size)
#print("total number of possible occlusions: ", len(givenOcclusions_stackImages))
#print(num_rows*num_columns)
#for i in range(len(givenOcclusions_stackImages)):
#    print("given Occlusion number %d stack images:"%i)
#    print(givenOcclusions_stackImages[i])
#    print(2*'\n')
#done!


# In[19]:


def patch_nonz_intersect(imgs, strides, patch_size):
    strides1 = strides[0] #stride along columns
    strides2 = strides[1] #stride along rows
    
    w  = patch_size[0]  #width of the patch
    h  = patch_size[1]  #height of the patch
    
    #
    n  = imgs.shape[0] # total number of images that belong to imgs
    n1 = imgs.shape[1] # the first dimension of each image
    n2 = imgs.shape[2] # the second dimension of each image
    
    
    iter_columns = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows    = range(0, n2-h+strides2, strides2) #iterator along rows object
    #total number of possible occlusions in each dimension 
    num_columns     = len(iter_columns) #total number of iterations along columns
    num_rows        = len(iter_rows)  #total number of iterations along rows
    total        = num_rows * num_columns 
    
    index_mat    = sub2ind_all([num_rows, num_columns])

         
    nonz_intersect_list  = [[] for _ in range(num_rows*num_columns)] # initializing a list for counting nonzero
    #intersection of each image with the patch at each location
    nonz_intersect_count = np.zeros((num_rows, num_columns), dtype = np.uint32) 
    #we want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for j in iter_columns:
        for i in iter_rows:

            if j == iter_columns[-1]:
                imgs_patch = imgs[:, i:i+h, j:n2+1].copy()

                for k in range(n):
                    nonz_intersect_size = imgs_patch[np.nonzero(imgs_patch[k])].size
                    nonz_intersect_list[index_mat[i,j]].append(nonz_intersect_size)
                    if nonz_intersect_size != 0:
                        nonz_intersect_count[i,j] += 1 
                #print(range(i, n2), range(j, j+h))
                #print(imgs_patch)
                #print(2*'\n')
                
            elif i == iter_rows[-1]:
                imgs_patch = imgs[:, i:n1, j:j+w].copy()
                
                for k in range(n):
                    nonz_intersect_size = imgs_patch[np.nonzero(imgs_patch[k])].size
                    nonz_intersect_list[index_mat[i,j]].append(nonz_intersect_size)
                    if nonz_intersect_size != 0:
                        nonz_intersect_count[i,j] += 1 
                #print(range(i, i+w), range(j,n1))
                #print(imgs_patch)
                #print(2*'\n')
                
            else:
                imgs_patch = imgs[:, i:i+h, j:j+w].copy()
                for k in range(n):
                    nonz_intersect_size = imgs_patch[np.nonzero(imgs_patch[k])].size
                    nonz_intersect_list[index_mat[i,j]].append(nonz_intersect_size)
                    if nonz_intersect_size != 0:
                        nonz_intersect_count[i,j] += 1 
                #print(range(i, i+w), range(j,j+w))
                #print(imgs_patch)
                #print(2*'\n')
                
    return (nonz_intersect_list, nonz_intersect_count)




#import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
#mnist0 = input_data.read_data_sets("MNIST_data/", one_hot=False)
#np.set_printoptions(threshold=np.nan)
#print(5*'\n')

#N                = 10000
#images           = mnist0.test.images[0:N]
#labels0          = mnist0.test.labels[0:N] #one_hot = False
#labels1          = mnist1.test.labels[0:N]


#patch_size    = [5, 5]
#strides       = [1, 1]

#for d in range(10):
#    d_indices = np.nonzero(labels0 == d)
    #print("digit %d indices: "%d)
    #print(d_indices)
    #print(2*'\n')
    #print("labels corresponding to digit %d: "%d)
    #print(list(labels0[d_indices]))
#    print("total number of digit %d labels: "%d)
#    print(len(list(labels0[d_indices])))
#    d_images  = images[d_indices].copy().reshape((-1,28,28))
    #print('\n')
#    print("shape of image %d array: "%d, d_images.shape)
#    (nonz_intersect_list, nonz_intersect_count) = patch_nonz_intersect(d_images , strides, patch_size)
#    print('\n')
#    print("nonzero intersection with the moving patch at each locations with images in class %d: "%d)
#    print(nonz_intersect_count)
#    print(2*'\n')
#    print("min interestion value for digit %d is: "%d, np.amin(nonz_intersect_count))
#    print(10*'\n')

#############################
#givenOcclusions_stackImages = fixOcclusions_stackImages(image_ones, strides, patch_size)
#print("total number of possible occlusions: ", len(givenOcclusions_stackImages))
#print(num_rows*num_columns)
#for i in range(len(givenOcclusions_stackImages)):
#    print("given Occlusion number %d stack images:"%i)
#    print(givenOcclusions_stackImages[i])
#    print(2*'\n')
#done!
###############################




