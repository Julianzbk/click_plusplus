#pragma once

#include "vectors.cuh"

#include <iostream>
#include <concepts>
#include <cassert>
#include <initializer_list>
#include <compare>
#include <functional>

#include <array>
#include <vector>

// CUDA API
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#pragma region bf16_operators
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
#pragma endregion bf16_operators


template <class dtype>
class DeviceVector
{
public:
    size_t size_;
    dtype* buf; // Device side

    DeviceVector()
        :size_(0), buf(nullptr)
    {
    }

    DeviceVector(std::initializer_list<dtype> const& V)
        :size_(V.size())
    {
        cudaMalloc(&buf, V.size() * sizeof(dtype));
        cudaMemcpy(buf, std::data(V), V.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    DeviceVector(std::vector<dtype> const& V)
        :size_(V.size())
    {
        cudaMalloc(&buf, V.size() * sizeof(dtype));
        cudaMemcpy(buf, V.data(), V.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    DeviceVector(DeviceVector const& A)
        :size_(A.size_)
    {
        cudaMalloc(&buf, A.size_ * sizeof(dtype));
        cudaMemcpy(buf, A.buf, A.size_ * sizeof(dtype), cudaMemcpyDeviceToDevice);
    }

    DeviceVector(DeviceVector && A)
        :size_(A.size_), buf(A.buf)
    {
    }

    static DeviceVector get_empty(size_t size)
    {// Factory function to prevent misuse
        DeviceVector<dtype> V;
        V.size_ = size;
        cudaMalloc(&V.buf, size * sizeof(dtype));
        return V;
    }

    ~DeviceVector()
    {
        cudaFree(buf);
    }

    std::vector<dtype> to_host() const
    {
        std::vector<dtype> host(size_);
        cudaMemcpy(host.data(), buf, size_ * sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }

    size_t size() const
    {
        return size_;
    }

    dtype* data() const
    {
        return buf;
    }

    dtype operator [] (size_t idx) const
    {// BAD
        dtype host;
        cudaMemcpy(&host, &buf[idx], sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }
};

template <typename dtype>
DeviceVector<dtype> to_device(std::vector<dtype> const& V)
{
    return DeviceVector(V);
}

template <typename T>
std::ostream& operator << (std::ostream& out, DeviceVector<T> V)
{
    if (V.size() <= 10)
    {
        T* host = new T[V.size()];
        cudaMemcpy(host, V.data(), V.size() * sizeof(T), cudaMemcpyDeviceToHost);

        out << '[';
        size_t i = 0;
        size_t size = V.size();
        for (; i < size - 1; ++i)
        {
            out << host[i] << ", ";
        }
        out << host[size - 1];
        out << ']';
        delete[] host;
    }
    else
    {
        out << "DeviceVector(" << V.size() << ")";
    }
    return out;
}


template <typename dtype, size_t M>
class DeviceArray
{
public:
    dtype* buf; // Device side

    DeviceArray()
        :buf()
    {
        cudaMalloc(&buf, M * sizeof(dtype));
    }

    DeviceArray(std::initializer_list<dtype> const& A)
    {
        assert(A.size() == M);
        cudaMalloc(&buf, M * sizeof(dtype));
        cudaMemcpy(buf, std::data(A), M * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    DeviceArray(std::array<dtype, M> const& A)
    {
        cudaMalloc(&buf, M * sizeof(dtype));
        cudaMemcpy(buf, A.data(), M * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    DeviceArray(DeviceArray const& A)
    {
        cudaMalloc(&buf, M * sizeof(dtype));
        cudaMemcpy(buf, A.buf, M * sizeof(dtype), cudaMemcpyDeviceToDevice);
    }

    DeviceArray(DeviceArray && A)
        :buf(A.buf)
    {
    }

    ~DeviceArray()
    {
        cudaFree(buf);
    }

    constexpr size_t size() const
    {
        return M;
    }

    dtype* data() const
    {
        return buf;
    }

    void fill(dtype val)
    {
        std::array<dtype, M> arr;
        arr.fill(val);
        cudaMemcpy(buf, arr.data(), M * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    std::array<dtype, M> to_host() const
    {
        std::array<dtype, M> host;
        cudaMemcpy(host.data(), buf, M * sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }

    dtype operator [] (size_t idx) const
    {// BAD
        dtype host;
        cudaMemcpy(&host, &buf[idx], sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }
};

template <typename dtype, size_t M>
DeviceArray<dtype, M> to_device(std::array<dtype, M> const& A)
{
    return DeviceArray(A);
}

template <typename T, size_t N>
std::ostream& operator << (std::ostream& out, DeviceArray<T, N> const& A)
{
    static_assert(N > 0);

    T host[N];
    cudaMemcpy(host, A.data(), N * sizeof(T), cudaMemcpyDeviceToHost);
    
    out << '[';
    size_t i = 0;
    for (; i < N - 1; ++i)
    {
        out << host[i] << ", ";
    }
    out << host[N - 1];
    out << ']';
    return out;
}


template <typename dtype, size_t M>
class DeviceMatrix
{
public:
    size_t size_;
    dtype* buf; // Device side

    DeviceMatrix(std::vector<std::array<dtype, M>> const& V)
        :size_(V.size())
    {
        cudaMalloc(&buf, V.size() * M * sizeof(dtype));
        dtype* dest = buf;
        for (std::array<dtype, M> x: V)
        {
            cudaMemcpy(dest, x.data(), x.size() * sizeof(dtype), cudaMemcpyHostToDevice);
            dest += M;
        }
    }

    DeviceMatrix(DeviceMatrix const& A)
    {
        cudaMalloc(&buf, A.size_ * M * sizeof(dtype));
        cudaMemcpy(buf, A.buf, A.size_ * M * sizeof(dtype), cudaMemcpyDeviceToDevice);
    }

    DeviceMatrix(DeviceMatrix && A)
        :size_(A.size_), buf(A.buf)
    {
    }
private:
    DeviceMatrix()
        :size_(0), buf(nullptr)
    {
    }
public:
    static DeviceMatrix<dtype, M> get_empty(size_t size)
    {// Factory function to prevent misuse
        DeviceMatrix<dtype, M> A;
        A.size_ = size;
        cudaMalloc(&A.buf, size * M * sizeof(dtype));
        return A;
    }

    ~DeviceMatrix()
    {
        cudaFree(buf);
    }

    size_t size() const
    {
        return size_;
    }

    size_t n_rows() const
    {
        return size_;
    }

    constexpr size_t n_columns() const
    {
        return M;
    }

    dtype* data() const
    {
        return buf;
    }

    DeviceArray<dtype, M> operator [] (size_t idx) const
    {
        DeviceArray<dtype, M> x;
        x.buf = buf + idx * M;
        return x;
    }
};

template <typename dtype, size_t M>
DeviceMatrix<dtype, M> to_device(std::vector<std::array<dtype, M>> const& V)
{
    return DeviceMatrix(V);
}

template <typename T, size_t N>
std::ostream& operator << (std::ostream& out, DeviceMatrix<T, N> const& M)
{
    static_assert(N > 0);
    out << "Matrix(" << N << ", " << M.size() << ")";
    return out;
}

#pragma region functions
template <typename dtype>
inline double scalar_log_loss_vector(dtype h, DeviceVector<dtype> const& Y)
{
    device::LogLoser<dtype> log_loser;
    log_loser.h = h;
    return device::vector_reduce<dtype, double, device::LogLoser<dtype>>(Y.data(), Y.size(), 0, log_loser);
}

template <typename dtype>
inline double vector_log_loss_vector(DeviceVector<dtype> const& H, DeviceVector<dtype> const& Y)
{
    assert(H.size() == Y.size());
    return device::vector_double_reduce<dtype, double>(H.data(), Y.data(), Y.size(), 0, device::log_loss);
}
#pragma endregion functions


template <typename dtype, size_t M>
DeviceArray<dtype, M> operator + (DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    DeviceArray<dtype, M> sum;
    sum.buf = device::vector_add_vector<dtype>(V.data(), U.data(), M);
    return sum;
}

template <typename dtype, size_t M>
DeviceArray<dtype, M>& operator += (DeviceArray<dtype, M>& V, DeviceArray<dtype, M> const& U)
{
    device::vector_addassign_vector<dtype>(V.data(), U.data(), M);
    return V;
}

template <typename dtype>
DeviceVector<dtype> operator - (DeviceVector<dtype> const& V, dtype A)
{
    DeviceVector<dtype> diff;
    diff.size_ = V.size();
    diff.buf = device::vector_sub_scalar<dtype>(V.data(), A, V.size());
    return diff;
}

template <typename dtype, size_t M>
DeviceArray<dtype, M> operator - (DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    DeviceArray<dtype, M> diff;
    diff.buf = device::vector_sub_vector<dtype>(V.data(), U.data(), M);
    return diff;
}

template <typename dtype, size_t M>
DeviceArray<dtype, M>& operator -= (DeviceArray<dtype, M>& V, DeviceArray<dtype, M> const& U)
{
    device::vector_subassign_vector<dtype>(V.data(), U.data(), M);
    return V;
}

template <typename itype, typename dtype, size_t M>
DeviceArray<dtype, M> operator * (itype A, DeviceArray<dtype, M> const& V)
{
    static_assert(!std::is_same_v<itype, DeviceArray<dtype, M>>);
    DeviceArray<dtype, M> U;
    U.buf = device::vector_mul_scalar<dtype>(V.data(), A, M);
    return U;
}

template <typename itype, typename dtype, size_t M>
DeviceArray<dtype, M> operator / (DeviceArray<dtype, M> const& V, itype A)
{
    static_assert(!std::is_same_v<itype, DeviceArray<dtype, M>>);
    DeviceArray<dtype, M> U;
    U.buf = device::vector_div_scalar<dtype>(V.data(), A, M);
    return U;
}

template <typename dtype>
inline dtype dot(DeviceVector<dtype> const& V, DeviceVector<dtype> const& U)
{
    assert(V.size() == U.size());
    return device::vector_dot<dtype>(V.data(), U.data(), V.size());
}

template <typename dtype, size_t M>
inline dtype dot(DeviceVector<dtype> const& V, DeviceArray<dtype, M> const& U)
{
    assert(V.size() == M);
    return device::vector_dot<dtype>(V.data(), U.data(), M);
}

template <typename dtype, size_t M>
inline dtype dot(DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    return device::vector_dot<dtype>(V.data(), U.data(), M);
}

template <typename dtype, size_t M>
DeviceVector<dtype> dot(DeviceArray<dtype, M> const& T,
                        DeviceMatrix<dtype, M> const& X,
                        dtype bias = 0)
{
    DeviceVector<dtype> Y;
    Y.size_ = X.size();
    Y.buf = device::vector_dot_matrix<dtype>(T.data(), X.data(), M, X.size(), bias);
    return Y;
}

template <typename dtype, size_t M, class DeviceLambda>
DeviceVector<dtype> dot_transform(DeviceArray<dtype, M> const& T,
                                  DeviceMatrix<dtype, M> const& X,
                                  dtype bias, DeviceLambda thunk)
{
    DeviceVector<dtype> Y;
    Y.size_ = X.size();
    Y.buf = device::vector_dot_matrix_transform<dtype>(T.data(), X.data(), M, X.size(), bias, thunk);
    return Y;
}

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
    
    size_t n_wrong = device::vector_double_reduce(pred.data(), Y.data(), pred.size(), 0ULL, ErrorFunctor<dtype>());
    return n_wrong / pred.size();
}