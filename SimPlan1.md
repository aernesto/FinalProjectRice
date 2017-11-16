# Simulation Plan 1

1. Train a convnet on MNIST                          
--> DONE
2. Save the trained network                        
--> DONE
3. Write function that occludes patch of an image.  
--> WE DECIDED TO USE ABSOLUTE PATCH POSITION ON FULL IMAGE 28x28
4. Select all images, in each class, from the test
set, that are correctly classified by the trained
network. 
5. Systematically compute the classification score
for each image and each patch location.
6. Compute average classification score for each
patch location (average across images in the class)
