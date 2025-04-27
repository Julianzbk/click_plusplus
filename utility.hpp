#pragma once

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <chrono>
#include <cstdint>

#ifdef USE_CUDA
    #include "vectors_device.hpp"
#endif

inline bool float_approx(double a, double b, double epsilon = 1e-9)
{
    return std::fabs(a - b) < epsilon;
}

template <typename T, typename Y, size_t M>
void zip_print(std::array<T, M> const& lhs, std::array<Y, M> const& rhs, std::ostream& out = std::cout)
{// Displays two arrays side by side. Useful for displaying datapoints.
    out << "[";
    for (size_t i = 0; i < M - 1; ++i)
    {
        out << lhs[i] << " = " << rhs[i] << ", ";
    }
    out << lhs[M - 1] << " = " << rhs[M - 1];
    out << "]";
}

template <typename itype = size_t>
class Rando
{
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

class Timer
{
public:
    std::chrono::time_point<std::chrono::high_resolution_clock> begin_time;

    void begin()
    {
        begin_time = std::chrono::high_resolution_clock::now();
    }

    void time_elapsed() const
    {
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - begin_time;
        std::cout << "\nTimer: " << duration << "\n" << std::endl;
    }

    void print_elapsed(std::string const& msg = "") const
    {
        std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - begin_time;
        std::cout << msg << " " << duration << "\n" << std::endl;
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

#ifdef USE_CUDA
template <typename dtype>
struct ErrorFunctor
{
    __device__
    size_t operator () (dtype pred, dtype Y) const
    {
        return (pred != Y) ? 1 : 0;
    }
};

template <typename dtype>
double error(DeviceVector<dtype> const& pred,
             DeviceVector<dtype> const& Y)
{
    size_t n_wrong = device::vector_double_reduce<dtype, size_t>(pred.data(), Y.data(), pred.size(), 0ULL, ErrorFunctor<dtype>());
    return (double) n_wrong / pred.size();
}

template <typename dtype>
double error(std::vector<dtype> const& pred,
             DeviceVector<dtype> const& Y)
{
    return error(pred, Y.to_host());
}

template <typename dtype>
double error(DeviceVector<dtype> const& pred,
             std::vector<dtype> const& Y)
{
    return error(pred.to_host(), Y);
}
#endif

struct ConfusionMatrix
{
    size_t true_negative = 0;
    size_t false_negative = 0;
    size_t false_positive = 0;
    size_t true_positive = 0;

    ConfusionMatrix& operator = (ConfusionMatrix const&) = default;

    friend std::ostream& operator << (std::ostream& out, ConfusionMatrix M)
    {
        size_t w0 = 2 + std::max(std::to_string(M.true_negative).length(), std::to_string(M.false_negative).length());
        size_t w1 = 2 + std::max(std::to_string(M.false_positive).length(), std::to_string(M.true_positive).length());
        out << "ConfusionMatrix:\n";
        out << "| " << std::left << std::setw(w0) << M.true_negative << std::right << std::setw(w1) << M.false_positive << " |\n";
        out << "| " << std::left << std::setw(w0) << M.false_negative << std::right << std::setw(w1) << M.true_positive << " |\n";
        return out;
    }

#ifdef USE_CUDA
    __host__ __device__
    ConfusionMatrix& operator += (ConfusionMatrix const& M)
    {// no-op
        return *this;
    }

    __host__ __device__
    friend ConfusionMatrix operator + (ConfusionMatrix const& lhs, ConfusionMatrix const& rhs)
    {// no-op
        return lhs;
    }
#endif
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

#ifdef USE_CUDA
static_assert(sizeof(size_t) == sizeof(unsigned long long));
template <typename dtype>
struct ConfusionFunctor
{
    unsigned long long* true_negative;
    unsigned long long* false_negative;
    unsigned long long* false_positive;
    unsigned long long* true_positive;

    ConfusionFunctor()
    {
        cudaMalloc(&true_negative, sizeof(unsigned long long) * 4);
        cudaMemset(true_negative, 0, sizeof(unsigned long long) * 4);
        false_negative = true_negative + 1;
        false_positive = true_negative + 2;
        true_positive = true_negative + 3;
    }
    
    __device__
    ConfusionMatrix operator () (dtype pred, dtype Y)
    {
        if (pred == 0 && Y == 0)
            atomicAdd(true_negative, 1ULL);
        else if (pred == 0 && Y == 1)
            atomicAdd(false_negative, 1ULL);
        else if (pred == 1 && Y == 0)
            atomicAdd(false_positive, 1ULL);
        else if (pred == 1 && Y == 1)
            atomicAdd(true_positive, 1ULL);
        return ConfusionMatrix();
    }

    __host__
    ConfusionMatrix export_() const
    {// kills the object.
        ConfusionMatrix M;
        cudaMemcpy(&M, true_negative, sizeof(unsigned long long) * 4, cudaMemcpyDeviceToHost);
        cudaFree(true_negative);
        return M;
    }
};

template <class dtype>
ConfusionMatrix confusion_matrix(std::vector<dtype> const& pred,
                                 DeviceVector<dtype> const& Y)
{
    return confusion_matrix(pred, Y.to_host()); // host is faster for larger vectors to avoid memcpy
}

template <class dtype>
ConfusionMatrix confusion_matrix(DeviceVector<dtype> const& pred,
                                 DeviceVector<dtype> const& Y)
{
    assert(pred.size() == Y.size());
    ConfusionFunctor<dtype> conf_functor;
    ConfusionMatrix dummy; // no-op accumulator
    device::vector_double_reduce<dtype, ConfusionMatrix>
        (pred.data(), Y.data(), pred.size(), dummy, conf_functor);
    return conf_functor.export_();
}
#endif

constexpr uint64_t FILE_MAGIC = 0x44432D4547464544;

template <typename T>
inline void dumps(std::ostream& out, T data)
{
    out.write(reinterpret_cast<const char*>(&data), sizeof(T));
}

template <typename T>
inline void dumps(std::ostream& out, T* data, size_t n_bytes)
{
    out.write(reinterpret_cast<const char*>(data), n_bytes);
}

template <typename T, typename file_T = char*>
inline T loads(std::istream& in)
{
    T data;
    in.read(reinterpret_cast<file_T>(&data), sizeof(T));
    return data;
}

template <typename T, typename file_T = char*>
inline void loads(std::istream& in, T* dest, size_t n_bytes)
{
    in.read(reinterpret_cast<file_T>(dest), n_bytes);
}

inline void word_assert(std::string word, std::string expect, std::string msg)
{
    if (word != expect)
    {
        std::cerr << msg << std::endl;
        throw std::runtime_error(msg);
    }
}

#ifdef USE_CUDA
#include <cuda_bf16.h>

// Inefficient host-side operators that casts to float first.
// If using dtype = bf16, the model should do all operations in CUDA device.
std::ostream& operator << (std::ostream& out, nv_bfloat16 bf)
{
    out << static_cast<float>(bf);
    return out;
}

auto operator <=> (nv_bfloat16 lhs, nv_bfloat16 rhs)
{
    return static_cast<float>(lhs) <=> static_cast<float>(rhs);
}

template <typename itype>
auto operator <=> (nv_bfloat16 lhs, itype rhs)
{
    static_assert(!std::is_same<itype, nv_bfloat16>::value);
    return static_cast<float>(lhs) <=> rhs;
}

template <typename itype>
nv_bfloat16 operator * (itype lhs, nv_bfloat16 rhs)
{
    static_assert(!std::is_same<itype, nv_bfloat16>::value);
    return lhs * static_cast<float>(rhs);
}

template <typename itype>
nv_bfloat16 operator / (nv_bfloat16 lhs, itype rhs)
{
    static_assert(!std::is_same<itype, nv_bfloat16>::value);
    return static_cast<float>(lhs), rhs;
}
#endif