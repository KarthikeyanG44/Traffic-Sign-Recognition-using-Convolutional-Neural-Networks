<h1>Traffic-Sign-Recognition-using-Convolutional-Neural-Networks</h1>
In this project, we explore the use of Spatial Transformer Networks interlaced between Convolutional Layers for more robust recognition.

CNN Architecture : 
STN -> Convolution ( 5X5 ) -> STN -> Convolution ( 3X3 ) -> STN -> Convolution ( 3X3 ) -> FC (350) -> FC(43)

STN Architecture: 
Localization Network + Sampling Grid Regressor
Convolutional ( 3X3 ) -> Convolutional ( 2X2 ) -> FC(32) -> FC(6)

Codes :
1. TrafficNet.py : Base file implementing the above mentioned architecture
2. data.py : File to apply various geometric distortions to augment the training dataset. NOTE : The same transformations must be applied on test data as well 
3. Train.py : Perform distributed training on a GPU enabled device with varying hyperparameter values
4. Test.py : Perform inference on CPU for the test data with the desired model
5. Visualize_STN.py : Visualize the effect of STN layers by sampling an image with the regressed sampling grid parameters.
