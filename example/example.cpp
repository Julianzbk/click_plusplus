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
    //cout << model.theta_ << " + " << model.bias_ << endl;
    DeviceDataset<dtype, M> valid("../train.csv", "click", {"id"}, rando.generate(), rando.generate() - 1000000);
    auto scaler = StandardScaler<dtype, M>::import_file("scaler.txt");
    scaler.transform(valid.X);
    auto pred = model.predict_device(valid.X);
    cout << "\nError rate: " << error(pred, valid.Y) << endl;
    cout << confusion_matrix(pred, valid.Y) << endl;

    size_t rand_i = Rando<size_t>(valid.n_rows).generate();
    cout << "Random datapoint: i = " << rand_i << endl;
    valid.print_point(rand_i);
    auto rand_x = valid.X[rand_i];
    cout << "Probability of click: " << model.predict_proba(rand_x) << endl;
    cout << "Prediction: " << model.predict(rand_x) << endl;
}