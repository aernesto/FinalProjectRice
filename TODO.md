1. Obtain accuracy score of trained network for each class in test set (with no occlusion)
2. Obtain accuracy score of trained network for each class, and for each patch location.
3. Calculate global accuracy on full (new) test set, for each patch location.
3. For each digit/class, and for each heatmap, get the minimum and maximum value, 
AND the location of the pixels with minimum and maximum value.
4. Produce a histogram of accuracies for each heatmap (only use the 5x5 patchsize)
5. Select, by inspection, what image from the test set (one per class) we use to examplify patch visualization.


For report:  
dscribe network architecture  
describe heatmap construction and meaning.


Separate point:  
Include ALL (or part of) occluded images in training set, and train afresh the network and redo full analysis so far.
