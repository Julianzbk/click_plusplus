#pragma once

#include <iostream>
#include <concepts>

#pragma region containers
#include <array>
template <typename T, size_t N>
std::ostream& operator << (std::ostream& out, std::array<T, N> const& v)
{
    static_assert(N > 0);
    out << '[';
    size_t i = 0;
    for (; i < N - 1; ++i)
    {
        out << v[i] << ", ";
    }
    out << v[N - 1];
    out << ']';
    return out;
}

#include <vector>
template <typename T>
std::ostream& operator << (std::ostream& out, std::vector<T> v)
{
    out << '[';
    int i = 0;
    int size = v.size();
    for (; i < size - 1; ++i)
    {
        out << v[i] << ", ";
    }
    out << v[size - 1];
    out << ']';
    return out;
}
#pragma endregion containers

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

#include <compare>
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


extern "C" float dot_product_vector_float(const float* A, const float* B, size_t N);


// Overloading operators for only scalar operations, or simple operations with the same type.
template <typename dtype>
std::vector<dtype> operator + (std::vector<dtype> const& V, dtype A)
{
    std::vector<dtype> sum(V.size());
    for (size_t i = 0; i < V.size(); ++i)
    {
        sum[i] = V[i] + A;
    }
    return sum;
}

template <typename dtype, size_t M>
std::array<dtype, M> operator + (std::array<dtype, M> const& V, std::array<dtype, M> const& U)
{
    std::array<dtype, M> sum;
    for (size_t i = 0; i < M; ++i)
    {
        sum[i] = V[i] + U[i];
    }
    return sum;
}

template <typename dtype, size_t M>
std::array<dtype, M>& operator += (std::array<dtype, M>& V, std::array<dtype, M> const& U)
{
    for (size_t i = 0; i < M; ++i)
    {
        V[i] += U[i];
    }
    return V;
}

template <typename dtype>
std::vector<dtype> operator - (std::vector<dtype> const& V, dtype A)
{
    std::vector<dtype> diff(V.size());
    for (size_t i = 0; i < V.size(); ++i)
    {
        diff[i] = V[i] - A;
    }
    return diff;
}

template <typename dtype, size_t M>
std::array<dtype, M> operator - (std::array<dtype, M> const& V, std::array<dtype, M> const& U)
{
    std::array<dtype, M> sum;
    for (size_t i = 0; i < M; ++i)
    {
        sum[i] = V[i] - U[i];
    }
    return sum;
}

template <typename dtype, size_t M>
std::array<dtype, M>& operator -= (std::array<dtype, M>& V, std::array<dtype, M> const& U)
{
    for (size_t i = 0; i < M; ++i)
    {
        V[i] -= U[i];
    }
    return V;
}

template <typename itype, typename dtype, size_t M>
std::array<dtype, M> operator * (itype A, std::array<dtype, M> const& V)
{
    static_assert(!std::is_same<itype, std::array<dtype, M>>::value);
    std::array<dtype, M> U;
    for (size_t i = 0; i < M; ++i)
    {
        U[i] = V[i] * A;
    }
    return U;
}

template <typename itype, typename dtype, size_t M>
std::array<dtype, M> operator / (std::array<dtype, M> const& V, itype A)
{
    static_assert(!std::is_same<itype, std::array<dtype, M>>::value);
    std::array<dtype, M> U;
    for (size_t i = 0; i < M; ++i)
    {
        U[i] = V[i] / A;
    }
    return U;
}


template <typename dtype>
inline dtype dot(std::vector<dtype> const& V, std::vector<dtype> const& U)
{
    assert(V.size() == U.size());
    return dot_product_vector_float(V.data(), U.data(), V.size());
}

template <typename dtype, size_t M>
inline dtype dot(std::vector<dtype> const& V, std::array<dtype, M> const& U)
{
    assert(V.size() == U.size());
    return dot_product_vector_float(V.data(), U.data(), V.size());
}

template <typename dtype, size_t M>
inline dtype dot(std::array<dtype, M> const& V, std::array<dtype, M> const& U)
{
    return dot_product_vector_float(V.data(), U.data(), M);
}

template <typename dtype, size_t M>
std::vector<dtype> dot(std::array<dtype, M> const& T,
                        std::vector<std::array<dtype, M>> const& X,
                        dtype bias = dtype())
{
    size_t N = X.size();
    std::vector<dtype> Y(N);
    for (size_t i = 0; i < N; ++i)
    {
        Y[i] = dot(X[i], T) + bias;
    }
    return Y;
}