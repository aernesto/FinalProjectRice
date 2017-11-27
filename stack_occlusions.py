
# coding: utf-8

# In[7]:


import numpy as np
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


# In[19]:


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

def fixImages_stackOcclusions(imgs, strides, patch_size):
    strides1 = strides[0] #stride along columns
    strides2 = strides[1] #stride along rows
    
    w  = patch_size[0]  #width of the patch
    h  = patch_size[1]  #height of the patch
    
    #
    n  = imgs.shape[0] # total number of images that belong to imgs
    n1 = imgs.shape[1] # the first dimension of each image
    n2 = imgs.shape[2] # the second dimension of each image
    
    
    GivenImages_stackOcclusions = [] #initializing a list for storing occluded images 
    
    iter_columns = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows    = range(0, n2-h+strides2, strides2) #iterator along rows object

    #total number of possible occlusions in each dimension 
    num_rows     = len(iter_columns) #total number of iterations along columns
    num_columns  = len(iter_rows)  #total number of iterations along rows


    #we want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for i in iter_columns:
        for j in iter_rows:

            occluded_imgs = imgs.copy()
            if i == iter_columns[-1]:
                occluded_imgs[:,i:n2+1, j:j+h] = 0
                #print(range(i, n2), range(j, j+h))
                #print(occluded_imgs)
                #print(2*'\n')
                GivenImages_stackOcclusions.append(occluded_imgs)
                
            elif j == iter_rows[-1]:
                occluded_imgs[:,i:i+w, j:n1+1] = 0
                #print(range(i, i+w), range(j,n1))
                #print(occluded_imgs)
                #print(2*'\n')
                GivenImages_stackOcclusions.append(occluded_imgs)
            else:
                occluded_imgs[:,i:i+w, j:j+h] = 0
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
    print("stack_occlusions_array's shape: ", stack_occlusions_array.shape) 
    return stack_occlusions_array

#stack_occlusions_array = fixImages_stackOcclusions(imgs, strides, patch_size)
#print("stack_occlusions_array")
#print(stack_occlusions_array)


# In[20]:


#fixOcclusions_stackImages
# but given an occlusion located in a particular area in the image, for all images in our digitClass
#we want to list that particular occlusion for all the images in our digitClass

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
    
    givenOcclusions_stackImages = [] #initializing a list for storing occluded images 
    
    iter_columns = range(0, n1-w+strides1, strides1) #iterator along columns object 
    iter_rows    = range(0, n2-h+strides2, strides2) #iterator along rows object
    #total number of possible occlusions in each dimension 
    num_rows     = len(iter_columns) #total number of iterations along columns
    num_columns  = len(iter_rows)  #total number of iterations along rows


    #we want the moving patch to act simultaneously
    #on all the images that belong to imgs at once:
    for i in iter_columns:
        for j in iter_rows:

            occluded_imgs = imgs.copy()
            if i == iter_columns[-1]:
                occluded_imgs[:,i:n2+1, j:j+h] = 0
                #print(range(i, n2), range(j, j+h))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages.append(occluded_imgs)
                
            elif j == iter_rows[-1]:
                occluded_imgs[:,i:i+w, j:n1+1] = 0
                #print(range(i, i+w), range(j,n1))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages.append(occluded_imgs)
            else:
                occluded_imgs[:,i:i+w, j:j+h] = 0
                #print(range(i, i+w), range(j,j+w))
                #print(occluded_imgs)
                #print(2*'\n')
                givenOcclusions_stackImages.append(occluded_imgs)
    return givenOcclusions_stackImages

#givenOcclusions_stackImages = fixOcclusions_stackImages(imgs, strides, patch_size)
#print(len(givenOcclusions_stackImages))
#print(num_rows*num_columns)
#print(givenOcclusions_stackImages)
#done!

