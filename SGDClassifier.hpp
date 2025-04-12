#pragma once

#include <iostream>
#include <format>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cassert>

#ifdef USE_DEVICE
    #include "vectors_device.hpp"
    using bf16 = nv_bfloat16;
#else
    #include "vectors_host.hpp"
    // using bf16 = std::bfloat16_t;
#endif


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

template <class dtype>
double error(std::vector<dtype> const& pred,
             std::vector<dtype> const& Y)
{
    assert(pred.size() == Y.size());
    size_t n_wrong = 0;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        if (pred[i] != Y[i])
            ++n_wrong;
    }
    return static_cast<double>(n_wrong) / pred.size();
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
    dtype bias_;
    float lr;
    float lambda;
    uint32_t max_epochs;
    bool early_stop;
    static constexpr uint32_t CALC_LOSS_EVERY = 1;
    static constexpr uint32_t PRINT_LOSS_EVERY = 1; // should be a multiple of CALC_LOSS_EVERY,
                                                    // set to 0 to disable print.
    static constexpr double CONVERGED_THRESH = 1e-4;

    SGDClassifier(float lr = 0.01, float lambda = 0.01, uint32_t max_epochs = 100, bool early_stop = false)
        :theta_(std::array<dtype, M>()), bias_(dtype()),
         early_stop(early_stop), lr(lr), max_epochs(max_epochs)
    {
    }

    static inline dtype sigmoid(dtype z)
    {
        return 1 / (1 + std::exp(-z));
    }

    static double loss(dtype h, dtype y)
    {
        return -(y * log(h + 1e-15) + (1 - y) * log(1 - h + 1e-15));
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
            // std::cout << "i = " << i << "\n";
            // for (size_t i = 0; i < X.size(); ++i) // Use for loop instead to train on full dataset.
            {
                std::array<dtype, M> xi = X[i];
                xi[0] = 0;
                dtype yi = Y[i];
                dtype z = dot(xi, theta_) + bias_;
                dtype h = sigmoid(z);
                dtype error = h - yi;
                std::array<dtype, M> grad = error * xi; // scalar mult with an array here.
                std::cout << "\npre-theta: " << theta_ << std::endl;
                std::array<dtype, M> pena = lambda * theta_;
                std::cout << "lambda: " << lambda << "\npenalty: " << pena << std::endl;
                theta_ -= lr * (grad + pena); // another scalar mult.
                std::cout << "post-theta: " << theta_ << std::endl;

                bias_ -= lr * error;
                
                for (dtype t: theta_)
                {
                    if (std::isinf(t))
                    {
                        std::cout << "inf! yi: " << yi << "\nxi:" << xi << std::endl;
                        std::cout << "error: " << error << "\ngrad:\n" << grad << std::endl;
                        return;
                    }
                    if (std::isnan(t))
                    {
                        std::cout << "nan! yi:\n" << yi << "\nh:\n" << h << std::endl;
                        std::cout << "z:\n" << h << "\nbias_:\n" << bias_ << std::endl;
                        return;
                    }
                }
            }
            
            if (epoch % CALC_LOSS_EVERY == 0)
            {
                std::cout << "pre matmul" << std::endl;
                std::vector<dtype> Z = dot(theta_, X);
                for (size_t i = 0; i < Z.size(); ++i)
                {
                    Z[i] = sigmoid(Z[i] + bias_);
                }
                loss = this->loss(Z, Y);
                
                if (PRINT_LOSS_EVERY > 0 && epoch % PRINT_LOSS_EVERY == 0)
                    std::cout << std::format("Epoch {}: Loss = {:.10f}\n", epoch, loss);

                if (early_stop &&
                    float_approx(last_losses[0], loss, CONVERGED_THRESH) &&
                    float_approx(last_losses[1], loss, CONVERGED_THRESH))
                {
                    std::cout << "Loss has converged!" << std::endl;
                    ++epoch;
                    break;
                }
                last_losses[1] = last_losses[0];
                last_losses[0] = loss;
            }
        }   
        std::cout << std::format("Training has ended after {} Epochs with Loss = {:.10f}.", epoch - 1, loss) << std::endl;
    }

    void partial_fit(std::array<dtype, M> const& x, dtype const& y)
    {
        dtype z = dot(x, theta_) + bias_;
        dtype h = sigmoid(z);
        dtype error = h - y;
        std::array<dtype, M> grad = error * x; // scalar mult with an array here.
        theta_ -= lr * grad; // another scalar mult.
        bias_ -= lr * error;
            
        if (PRINT_LOSS_EVERY > 0)
        {
            dtype z = dot(theta_, x);
            z = sigmoid(z + bias_);
            std::cout << std::format("Partial-fit: Loss = {:.5f}\n", loss(z, y));
        }
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

void test_example()
{
    /*
    using bf16 = nv_bfloat16;
    using std::cout, std::endl;
    auto model = SGDClassifier<bf16, 2>(0.1, 0.01, 1000);
    std::vector<std::array<bf16, 2>> X_tr(4);
    X_tr[0] = std::array<bf16, 2>({1.0, 2.0});
    X_tr[1] = std::array<bf16, 2>({0.25, 0.2});
    X_tr[2] = std::array<bf16, 2>({0.0, 0.0});
    X_tr[3] = std::array<bf16, 2>({0.6, 1.25});
    std::vector<bf16> y_tr = {1, 0, 0, 1};
    model.fit(X_tr, y_tr);
    cout << model.theta_ << ", " << model.bias_ << endl;
    cout << model.predict({{2, 1}, {0, -1}, {0, 3}}) << endl;
    */
}