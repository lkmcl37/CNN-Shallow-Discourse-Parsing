This project aims to build discourse relation classifiers for gold argument pairs in PDTB with neural network approaches. 

This project implements two kinds of neural networks:

#Two-CNN Text Relation Classifier
Structure: 
1. use one embedding layer to transform two arguments into two different fixed dimension vectors; 
2. use two different convolutional layers to extract features of each argument respectively; 
3. feed each feature produced to a Batch Normalization layers and output new features; 
4. concat the features of two arguments; 5. use softmax to perform discourse classification on the arguments. 
*L2 regularization is applied to prevent overfitting. 

#Two-LSTM Text Relation Classifier

The structure is exactly the same as two-CNN’s, except the convolutional layers are replaced by Bi-directional LSTM cells, and there is no batch normalization layer.
