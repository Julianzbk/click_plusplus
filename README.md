# click_plusplus
Click-through rate predictor implemented in C++, powered by CUDA-accelerated computations library made from-scratch. <br>
Skills explored: **Logistic Regression**, **CUDA kernels**, **Gradient Descent**, Impl of Pytorch, Linear Regression, Impl of Chain Rule, Linear Algebra. <br>
Frameworks used: C++, CUDA, Tableau, GTest.

# Quick demo
Unzip profile.zip and run profile.bat, which trains a classifier model on the train.csv dataset, and assesses its prediction accuracy. Time elapsed in each operation is measured, compare host and device performance and decide whether to use hardware accelerations or not. <br><br>
```example/example.exe``` loads a pre-trained model, complete with source file ```example.cpp``` that demonstrate model and library interface. <br>

# How to use this model
1. ```#include "SGDClassifier.hpp"```, which includes all other necessary headers. <br>
    a. ```#define USE_CUDA``` to enable CUDA acceleration, this will replace API of STL containers with that of device-side containers.
2. Instantiate a instance of ```SGDClassifier<typename dtype, size_t M>```, where ```dtype``` is the datatype used in the training dataset, and ```M``` the number of its input features.
3. Train the model with ```SGDClassifier::fit(Matrix, Vector)```. <br>
    a. With no CUDA, ```Matrix``` is defined as ```std::vector<std::array<dtype, M>>```, and ```Vector``` is simply ```std::vector<dtype>``` <br>
    b. With CUDA enabled, the model instead uses ```DeviceMatrix``` and ```DeviceVector```, but stil supports STL arguments. <br>
4. Do regression with ```Vector SGDClassifier::predict(Matrix)```, which produces binary outputs, or ```predict_proba```, which produces probabilities of the output being 1.

# Custom library
Header-only libraries that introduce a myriad of generic-compatible Data structures, algorithms, and utility.

### SGDClassifier.hpp
A linear regression classifier that uses a logistic sigmoid activation function to produce probabilistic outputs. The linear weights are trained via Stochastic Gradient Descent with L1 regularization. The loss curve is calculated at every epoch to enable convergence early-stopping.

### vectors_device.hpp
Create vectors and matrices that live on the GPU, and run any algorithms in parallel with blazing speed.

### dataset.hpp
Read a single csv file into feature (X) and target (Y) vectors. Skip the features you don't want, transform select features, and normalize the dataset.
