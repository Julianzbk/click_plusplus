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
dtype dot(std::vector<dtype> const& V, std::vector<dtype> const& U)
{
    dtype acc = dtype();
    for (size_t i = 0; i < V.size(); ++i)
    {
        acc += V[i] * U[i];
    }
    return acc;
}

template <typename dtype, size_t M>
dtype dot(std::vector<dtype> const& V, std::array<dtype, M> const& U)
{
    dtype acc = dtype();
    for (size_t i = 0; i < M; ++i)
    {
        acc += V[i] * U[i];
    }
    return acc;
}

template <typename dtype, size_t M>
dtype dot(std::array<dtype, M> const& V, std::array<dtype, M> const& U)
{
    dtype acc = dtype();
    for (size_t i = 0; i < M; ++i)
    {
        acc += V[i] * U[i];
    }
    return acc;
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

/*
template <typename dtype>
std::vector<dtype> matmul(std::vector<std::vector<dtype>> const& A,
                          std::vector<std::vector<dtype>> const& X)
{
    std::vector<dtype> B(d);
    for (std::vector<dtype> i: A)
    {
        for (std::vector<dtype> j: X)
        {
            acc = V * u;
        }
    }
    return acc;
}
*/