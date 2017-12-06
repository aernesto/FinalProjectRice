1. Obtain accuracy score of trained network for each class in test set (with no occlusion)
2. Obtain accuracy score of trained network for each class, and for each patch location.
3. Calculate global accuracy on full (new) test set, for each patch location.
3. For each digit/class, and for each heatmap, get the minimum and maximum value, 
AND the location of the pixels with minimum and maximum value.
4. Produce a histogram of accuracies for each heatmap (only use the 5x5 patchsize)
5. Select, by inspection, what image from the test set (one per class) we use to examplify patch visualization.
6. For each one of the following patchsize (2x2, 5x5, 8x8), do the following:  
- produce the histogram of accuracies (over all patch locations), with full test set (So all images from the test set are used, several times because each image is used with all the patch locations, one by one... all goes into the histogram). So, at the end, we have 3 histograms (one per patch size).  
- compute mean and std dev of each histogram.  


For report:  
dscribe network architecture  
describe heatmap construction and meaning.


Separate point:  
Include ALL (or part of) occluded images in training set, and train afresh the network and redo full analysis so far.
