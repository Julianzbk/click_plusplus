#define USE_CUDA
#include "SGDClassifier.hpp"
#include "dataset.hpp"

#include <iostream>
using std::cout, std::endl;

// Enter dimensions of dataset as macros:
#define N_FEATURES 22
#define N_FIELDS 24

int main()
{
    using dtype = float;
    constexpr size_t M = N_FEATURES;

    Rando<size_t> rando(1000000, 2000000);
    auto model = SGDClassifier<dtype, M>::import_file("weights.txt");
    DeviceDataset<dtype, M> valid("train.csv", "click", {"id"}, rando.generate(), rando.generate() - 1000000);
    StandardScaler scaler;
    scaler.fit(valid.X);
    scaler.transform(valid.X);
    cout << "Error rate: " << error(model.predict_device(valid.X), valid.Y) << endl;
}