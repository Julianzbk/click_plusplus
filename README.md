# click_plusplus
Click-through rate Regressor implemented in C++ with CUDA-accelerated computations. <br>
Skills explored: **Logistic Regression**, **CUDA kernels**, **Gradient Descent**, Linear Regression, Impl of Chain Rule, Linear Algebra. <br>
Frameworks used: C++, CUDA, Tableau, GTest

# How to use this model
1. ```#include "SGDClassifier.hpp"```, which includes all other necessary headers. <br>
    a. ```#define USE_CUDA``` to enable CUDA acceleration, this will replace API of STL containers with that of device-side containers.
2. Instantiate a instance of ```SGDClassifier<typename dtype, size_t M>```, where ```dtype``` is the datatype used in the training dataset, and ```M``` the number of its input features.
3. Train the model with ```SGDClassifier::fit(Matrix, Vector)```. <br>
    a. With no CUDA, ```Matrix``` is defined as ```std::vector<std::array<dtype, M>>```, and ```Vector``` is simply ```std::vector<dtype>``` <br>
    b. With CUDA enabled, the model instead uses ```DeviceMatrix``` and ```DeviceVector```, but stil supports STL arguments. <br>
4. Do regression with ```Vector SGDClassifier::predict(Matrix)```, which produces binary outputs, and ```predict_proba``, which produces probabilities of the output being 1.

# SGD Classifier
