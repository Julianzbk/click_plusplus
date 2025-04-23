#pragma once

#include <iostream>
#include <iomanip>
#include <format>
#include <cmath>
#include <numeric>
#include <random>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <fstream>

enum class Device {cpu, cuda};
// for control flow with if constexpr

#ifdef USE_CUDA
    #include "vectors_device.hpp"
    using namespace device;
    using bf16 = nv_bfloat16;
    constexpr Device DEVICE = Device::cuda;
    
#else
    #include "vectors_host.hpp"
    using namespace host;
    // using bf16 = std::bfloat16_t;
    constexpr Device DEVICE = Device::cpu;
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

struct ConfusionMatrix
{
    size_t true_negative = 0;
    size_t false_positive = 0;
    size_t false_negative = 0;
    size_t true_positive = 0;

    friend std::ostream& operator << (std::ostream& out, ConfusionMatrix M)
    {
        out << "ConfusionMatrix:\n";
        out << "| " << M.true_negative << "\t" << std::setw(4) << std::right << M.false_positive << "\n";
        out << "| " << M.false_negative << "\t" << std::setw(4) << std::right << M.true_positive << "\n";
        return out;
    }
};

template <class dtype>
ConfusionMatrix confusion_matrix(std::vector<dtype> const& pred,
                                 std::vector<dtype> const& Y)
{
    assert(pred.size() == Y.size());
    ConfusionMatrix conf;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        if (pred[i] == 0 && Y[i] == 0)
            ++conf.true_negative;
        else if (pred[i] == 0 && Y[i] == 1)
            ++conf.false_negative;
        else if (pred[i] == 1 && Y[i] == 0)
            ++conf.false_positive;
        else if (pred[i] == 1 && Y[i] == 1)
            ++conf.true_positive;
    }
    return conf;
}

constexpr uint64_t FILE_MAGIC = 0x44454647452D4344;

template <typename T>
inline void dumps(std::ostream& out, T data)
{
    out.write(reinterpret_cast<const char*>(&data), sizeof(data));
}

template <typename T>
inline void dumps(std::ostream& out, const T& data, size_t n_bytes)
{
    out.write(reinterpret_cast<const char*>(&data), n_bytes);
}

template <typename read_T>
inline read_T loads(std::istream& in)
{
    read_T data;
    in.read(reinterpret_cast<char*>(&data), sizeof(read_T));
    return data;
}

template <typename read_T>
inline void loads(std::istream& in, read_T* dest,size_t n_bytes)
{
    in.read(reinterpret_cast<read_T>(dest), n_bytes);
}
#pragma endregion utility

template <typename dtype, size_t M> 
class SGDClassifier
{
    /*
        Logistic Regression Classifier with Stochastic Gradient Descent.
    */
#ifdef USE_CUDA
    using Vector = DeviceVector<dtype>;
    using Array = DeviceArray<dtype, M>;
    using Matrix = DeviceMatrix<dtype, M>;
#else
    using Vector = std::vector<dtype>;
    using Array = std::array<dtype, M>;
    using Matrix = std::vector<std::array<dtype, M>>;
#endif
public:
    Array theta_;
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
        :theta_(Array()), bias_(dtype()), lr(lr), lambda(lambda),
         max_epochs(max_epochs), early_stop(early_stop)
    {
    }

    static inline dtype activ(dtype z)
    {
        return sigmoid(z);
    }

    static inline double loss(dtype h, dtype y)
    {
        return log_loss(h, y);
    }

    static double loss(dtype h, Vector const& Y)
    {
        return scalar_log_loss_vector(h, Y);
    }

    static double loss(Vector const& H, Vector const& Y)
    {
        return vector_log_loss_vector(H, Y);
    }

    void fit(Matrix const& X, Vector const& Y)
    {
    /*
        Required Interfaces:
            All math operators of dtype
            Array::fill
            Array Matrix::operator[]
            dtype Array:operator[]
            dtype Vector:operator[]
            dtype dot(Array, Array)
            sigmoid(dtype)
            dtype * Array
            Array + Array
            Array -= Array
    */
        assert(X.size() == Y.size());
        theta_.fill(0); // Zero out the weights
        
        Rando random_state(0, X.size());
        double last_losses[2] = {1.0, 1.0};
        double loss = 1.0;
        uint32_t epoch = 1;
        for (; epoch <= max_epochs; ++epoch)
        {
            size_t i = random_state.generate(); // Stoichastic = choose random datapoint from X.
            // for (size_t i = 0; i < X.size(); ++i) // Use for loop instead to train on full dataset.
            {
                Array xi = X[i];
                dtype yi = Y[i];
                dtype z = dot(xi, theta_) + bias_;
                dtype h = sigmoid(z);
                dtype error = h - yi;
                Array grad = error * xi; // scalar mult with an array here.
                Array penalty = lambda * theta_;
                
                //cout << "theta_: " << theta_.to_host() << " bias_: " << bias_ << "\nxi: " << xi.to_host() << endl;
                theta_ -= lr * (grad + penalty); // another scalar mult.
                bias_ -= lr * error;
            }
            
            if (epoch % CALC_LOSS_EVERY == 0)
            {
                Vector Z = dot_transform(theta_, X, bias_, sigmoid); // TODO change sigmoid to activ?
                loss = this->loss(Z, Y);
                
                if (PRINT_LOSS_EVERY > 0 && epoch % PRINT_LOSS_EVERY == 0)
                    std::cout << "Epoch " << epoch << ": Loss = " << std::setprecision(10) << loss << "\n";

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
        std::cout << "Training has ended after " << epoch - 1 << " Epochs with Loss = " << std::setprecision(10) << loss << std::endl;
    }

    void partial_fit(Array const& x, dtype const& y)
    {
        dtype z = dot(x, theta_) + bias_;
        dtype h = sigmoid(z);
        dtype error = h - y;
        Array grad = error * x; // scalar mult with an array here.
        theta_ -= lr * grad; // another scalar mult.
        bias_ -= lr * error;
            
        if (PRINT_LOSS_EVERY > 0)
        {
            std::cout << "Partial-fit: Loss = " << std::setprecision(5)  << loss(h, y) << "\n";
        }
    }

    Vector predict_proba(Matrix const& X)
    {
        return dot_transform(theta_, X, bias_, sigmoid);
    }

#ifdef USE_CUDA
    void fit(std::vector<std::array<dtype, M>> const& X,
             std::vector<dtype> const& Y)
    {
        fit(to_device(X), to_device(Y));
    }
    
    std::vector<dtype> predict_proba(std::vector<std::array<dtype, M>> const& X)
    {
        return predict_proba(to_device(X)).to_host();
    }

    std::vector<dtype> predict(std::vector<std::array<dtype, M>> const& X)
    {
        return predict(to_device(X));
    }

    struct ThresholdFunctor
    {
        __device__
        dtype operator () (dtype p) const
        {
            return (p >= 0.5) ? 1 : 0;
        }
    } thresh_functor;

    DeviceVector<dtype> predict_device(Matrix const& X)
    {
        DeviceVector pred = predict_proba(X);
        vector_inplace_transform<dtype, ThresholdFunctor>(pred.data(), pred.size(), 0, thresh_functor);
        return pred;
    } 
#endif
 
    std::vector<dtype> predict(Matrix const& X)
    {
    #ifdef USE_CUDA
        std::vector<dtype> proba = predict_proba(X).to_host();
    #else
        std::vector<dtype> proba = predict_proba(X);
    #endif
        std::transform(proba.cbegin(), proba.cend(),
                       proba.begin(), [](dtype p){return (dtype) (p >= 0.5 ? 1 : 0);});
        return proba;
    }

    void export_file(std::string const& path)
    {
        std::ofstream fout(path, std::ios::binary);
        if (!fout)
            throw std::runtime_error("Bad file read");
        dumps(fout, FILE_MAGIC);
        size_t n_bytes = theta_.size() * sizeof(dtype);
        dumps(fout, n_bytes);
        #ifdef USE_CUDA
            dumps(fout, theta_.to_host().data(), n_bytes);
        #else
            dumps(fout, theta_.data(), n_bytes);
        #endif
        dumps(fout, bias_);
    }

    static SGDClassifier import_file(std::string const& path)
    {
        std::ifstream fin(path, std::ios::binary);
        if (loads<uint64_t>(fin) != FILE_MAGIC)
            throw std::runtime_error("Bad file read, not formatted correctly");
        size_t n_bytes = loads<size_t>(fin);
        assert(M == (n_bytes / sizeof(dtype)));
        std::array<dtype, M> theta_;
        loads(fin, theta_.data(), n_bytes);
        SGDClassifier model;
        #ifdef USE_CUDA
            model.theta_ = to_device(theta_);   
        #else
            model.theta_ = theta_;
        #endif
        model.bias_ = loads<dtype>(fin);
        return model;
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
