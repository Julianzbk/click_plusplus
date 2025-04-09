#include <format>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <stdfloat>
#include <cstdint>
#include <cassert>

#include "vectors.hpp"

#include <iostream>
using std::cout, std::endl;

using bf16 = std::bfloat16_t;

#pragma region utility
inline bool float_approx(double a, double b, double epsilon = 1e-9)
{
    return std::fabs(a - b) < epsilon;
}

class Rando
{
    using itype = size_t;

    std::mt19937 gen;
    std::uniform_int_distribution<> dist;
public:
    Rando(itype upper)
        :dist(0, upper)
    {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    Rando(itype lower, itype upper)
        :dist(lower, upper)
    {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    Rando(itype lower, itype upper, uint32_t seed)
        :gen(seed), dist(lower, upper)
    {
    }
    
    itype generate()
    {
        return dist(gen);
    }
};

template <class dtype, size_t M>
double error(std::vector<dtype> const& pred,
             std::vector<dtype> const& Y)
{
    assert(pred.size() == Y.size());
    size_t n_correct = 0;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        if (pred[i] == Y[i])
            ++n_correct;
    }
    return static_cast<double>(n_correct) / pred.size();
}
#pragma endregion utility

template <class dtype, size_t M>
class SGDClassifier
{
    /*
        Logistic Regression Classifier with Stochastic Gradient Descent.
    */
    using Matrix = std::vector<std::array<dtype, M>>; // use if you'd like.
public:
    std::array<dtype, M> theta_;
    std::array<std::string, M> classes_;
    dtype bias_;
    float lr;
    uint32_t max_epochs;
    static constexpr uint32_t CALC_LOSS_EVERY = 1;
    static constexpr uint32_t PRINT_LOSS_EVERY = 5; // should be a multiple of CALC_LOSS_EVERY,
                                                    // set to 0 to disable print.
    static constexpr double CONVERGED_THRESH = 1e-4;

    SGDClassifier(float lr = 0.01, uint32_t max_epochs = 100)
        :theta_(std::array<dtype, M>()), bias_(dtype()),
         lr(lr), max_epochs(max_epochs)
    {
    }

    static inline dtype sigmoid(dtype z)
    {
        return 1 / (1 + std::exp(-z));
    }

    static double loss(dtype h, std::vector<dtype> const& Y)
    {
        double acc = 0.0;
        const double proba = log(h + 1e-15);
        const double proba_bar = log(1 - h + 1e-15);
        for (dtype y: Y)
        {
            acc += y * proba + (1 - y) * proba_bar;
        }
        return -acc / Y.size();
    }

    static double loss(std::vector<dtype> const& H,
                       std::vector<dtype> const& Y)
    {
        double acc = 0.0;
        for (size_t i = 0; i < H.size(); ++i)
        {
            acc += Y[i] * log(H[i] + 1e-15) + (1 - Y[i]) * log(1 - H[i] + 1e-15);
        }
        return -acc / Y.size();
    }

    void fit(std::vector<std::array<dtype, M>> const& X,
             std::vector<dtype> const& Y)
    {
        assert(X.size() == Y.size());
        // Zero out the weights
        for (size_t i = 0; i < M; ++i)
        {
            theta_[i] = 0;
        }
        
        Rando random_state(0, X.size());
        double last_losses[2] = {1.0, 1.0};
        double loss = 1.0;
        uint32_t epoch = 1;
        for (; epoch <= max_epochs; ++epoch)
        {
            size_t i = random_state.generate(); // Stoichastic = choose random datapoint from X.
            //for (size_t i = 0; i < X.size(); ++i) // Use for loop instead to train on full dataset.
            {
                std::array<dtype, M> xi = X[i];
                dtype yi = Y[i];
                dtype z = dot(xi, theta_) + bias_;
                dtype h = sigmoid(z);
                dtype error = h - yi;
                std::array<dtype, M> grad = error * xi; // scalar mult with an array here.
                theta_ -= lr * grad; // another scalar mult.
                bias_ -= lr * error;
            }
            
            if (epoch % CALC_LOSS_EVERY == 0)
            {
                std::vector<dtype> Z = dot(theta_, X);
                for (size_t i = 0; i < Z.size(); ++i)
                {
                    Z[i] = sigmoid(Z[i] + bias_);
                }
                loss = this->loss(Z, Y);

                if (PRINT_LOSS_EVERY > 0 && epoch % PRINT_LOSS_EVERY == 0)
                    std::cout << std::format("Epoch {}: Loss = {:.5f}\n", epoch, loss);

                if (float_approx(last_losses[0], loss, CONVERGED_THRESH) &&
                    float_approx(last_losses[1], loss, CONVERGED_THRESH))
                {
                    std::cout << "Loss has converged!" << endl;
                    ++epoch;
                    break;
                }
                last_losses[1] = last_losses[0];
                last_losses[0] = loss;
            }
        }
        std::cout << std::format("Training has ended after {} Epochs with Loss = {:.5f}.", epoch - 1, loss) << endl;
    }

    std::vector<dtype> predict_proba(std::vector<std::array<dtype, M>> const& X)
    {
        std::vector<dtype> Z = dot(theta_, X);
        for (size_t i = 0; i < Z.size(); ++i)
        {
            Z[i] = sigmoid(Z[i] + bias_);
        }
        return Z;
    }

    std::vector<dtype> predict(std::vector<std::array<dtype, M>> const& X)
    {
        auto proba = predict_proba(X);
        std::transform(proba.cbegin(), proba.cend(),
                       proba.begin(), [](dtype p){return p >= 0.5 ? 1 : 0;});
        return proba;
    }
};

int test_example()
{
    auto model = SGDClassifier<bf16, 2>(0.1, 1000);
    std::vector<std::array<bf16, 2>> X_tr(4);
    X_tr[0] = std::array<bf16, 2>({1.0, 2.0});
    X_tr[1] = std::array<bf16, 2>({0.25, 0.2});
    X_tr[2] = std::array<bf16, 2>({0.0, 0.0});
    X_tr[3] = std::array<bf16, 2>({0.6, 1.25});
    std::vector<bf16> y_tr = {1, 0, 0, 1};
    model.fit(X_tr, y_tr);
    cout << model.theta_ << ", " << model.bias_ << endl;
    cout << model.predict({{2, 1}, {0, -1}, {0, 3}}) << endl;
}