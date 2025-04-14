#pragma once

#include <iostream>
#include <concepts>
#include <cassert>
#include <initializer_list>

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


extern "C" float dot_vector_float(const float* A, const float* B, size_t N);


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
    std::array<dtype, M> diff;
    for (size_t i = 0; i < M; ++i)
    {
        diff[i] = V[i] - U[i];
    }
    return diff;
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
    if constexpr (std::is_same_v<dtype, float>)
        return dot_vector_float(V.data(), U.data(), V.size());
    else
        std::cout << "dot: dtype not implemented!" << std::endl;
}

template <typename dtype, size_t M>
inline dtype dot(std::vector<dtype> const& V, std::array<dtype, M> const& U)
{
    assert(V.size() == U.size());
    if constexpr (std::is_same_v<dtype, float>)
        return dot_vector_float(V.data(), U.data(), M);
    else
        std::cout << "dot: dtype not implemented!" << std::endl;
}

template <typename dtype, size_t M>
inline dtype dot(std::array<dtype, M> const& V, std::array<dtype, M> const& U)
{
    if constexpr (std::is_same_v<dtype, float>)
        return dot_vector_float(V.data(), U.data(), M);
    else
        std::cout << "dot: dtype not implemented!" << std::endl;
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



template <class dtype>
class DeviceVector
{
public:
    size_t size;
    dtype* buf; // Device side

    DeviceVector()
        :size(0), buf(nullptr)
    {
    }

    DeviceVector(std::initializer_list<dtype> const& V)
        :size(V.size())
    {
        cudaMalloc(&buf, V.size() * sizeof(dtype));
        cudaMemcpy(buf, std::data(V), V.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    DeviceVector(std::vector<dtype> const& V)
        :size(V.size())
    {
        cudaMalloc(&buf, V.size() * sizeof(dtype));
        cudaMemcpy(buf, V.data(), V.size() * sizeof(dtype), cudaMemcpyHostToDevice);
    }

    ~DeviceVector()
    {
        cudaFree(buf);
    }

    dtype operator [] (size_t idx) const
    {// BAD
        std::cout << "BAD!" << std::endl;
        dtype host;
        cudaMemcpy(&host, &buf[idx], sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }
};

template <typename T>
std::ostream& operator << (std::ostream& out, DeviceVector<T> V)
{
    T* host = new T[V.size];
    cudaMemcpy(host, V.buf, V.size * sizeof(T), cudaMemcpyDeviceToHost);

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
    return out;
}


template <class dtype, size_t M>
class DeviceArray
{
public:
    dtype* buf; // Device side

    DeviceArray()
        :buf(nullptr)
    {
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

    ~DeviceArray()
    {
        cudaFree(buf);
    }

    constexpr size_t size() const
    {
        return M;
    }

    dtype operator [] (size_t idx) const
    {// BAD
        dtype host;
        cudaMemcpy(&host, &buf[idx], sizeof(dtype), cudaMemcpyDeviceToHost);
        return host;
    }
};

template <typename T, size_t N>
std::ostream& operator << (std::ostream& out, DeviceArray<T, N> const& A)
{
    static_assert(N > 0);

    T host[N];
    cudaMemcpy(host, A.buf, N * sizeof(T), cudaMemcpyDeviceToHost);
    
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


template <class dtype, size_t M>
class DeviceMatrix
{
public:
    size_t size;
    dtype* buf; // Device side

    DeviceMatrix(std::vector<std::array<dtype, M>> const& V)
        :size(V.size())
    {
        cudaMalloc(&buf, V.size() * M * sizeof(dtype));
        dtype* dest = buf;
        for (std::array<dtype, M> x: V)
        {
            cudaMemcpy(dest, x.data(), x.size() * sizeof(dtype), cudaMemcpyHostToDevice);
            dest += M;
        }
    }

    size_t N() const
    {
        return size;
    }

    DeviceArray<dtype, M> operator [] (size_t idx) const
    {
        DeviceArray<dtype, M> x;
        x.buf = buf + idx * M;
        return x;
    }
};

template <typename T, size_t N>
std::ostream& operator << (std::ostream& out, DeviceMatrix<T, N> const& M)
{
    static_assert(N > 0);

    out << "Matrix(" << N << ", " << M.size << ")";
    return out;
}

extern "C" float* vector_add_vector_float(const float* V, const float* U, size_t N);

template <typename dtype, size_t M>
DeviceArray<dtype, M> operator + (DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    DeviceArray<dtype, M> sum;
    if constexpr (std::is_same_v<dtype, float>)
        sum.buf = vector_add_vector_float(V.buf, U.buf, M);
    else
        std::cout << "operator + : dtype not implemented!" << std::endl;
    return sum;
}

extern "C" void vector_addassign_vector_float(float* V, const float* U, size_t N);

template <typename dtype, size_t M>
DeviceArray<dtype, M>& operator += (DeviceArray<dtype, M>& V, DeviceArray<dtype, M> const& U)
{
    if constexpr (std::is_same_v<dtype, float>)
        vector_addassign_vector_float(V.buf, U.buf, M);
    else
        std::cout << "operator + : dtype not implemented!" << std::endl;
    return V;
}

extern "C" float* vector_sub_scalar_float(const float* V, float A, size_t N);

template <typename dtype>
DeviceVector<dtype> operator - (DeviceVector<dtype> const& V, dtype A)
{
    DeviceVector<dtype> diff;
    if constexpr (std::is_same_v<dtype, float>)
        diff.buf = vector_sub_scalar_float(V.buf, A, V.size);
    else
        std::cout << "operator - : dtype not implemented!" << std::endl;
    return diff;
}

extern "C" float* vector_sub_vector_float(const float* V, const float* U, size_t N);

template <typename dtype, size_t M>
DeviceArray<dtype, M> operator - (DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    DeviceArray<dtype, M> diff;
    if constexpr (std::is_same_v<dtype, float>)
        diff.buf = vector_sub_vector_float(V.buf, U.buf, M);
    else
        std::cout << "operator - : dtype not implemented!" << std::endl;
    return diff;
}

extern "C" void vector_subassign_vector_float(float* V, const float* U, size_t N);

template <typename dtype, size_t M>
DeviceArray<dtype, M>& operator -= (DeviceArray<dtype, M>& V, DeviceArray<dtype, M> const& U)
{
    if constexpr (std::is_same_v<dtype, float>)
        vector_subassign_vector_float(V.buf, U.buf, M);
    else
        std::cout << "operator -= : dtype not implemented!" << std::endl;
    return V;
}

extern "C" float* vector_mul_scalar_float(const float* V, float A, size_t N);

template <typename itype, typename dtype, size_t M>
DeviceArray<dtype, M> operator * (itype A, DeviceArray<dtype, M> const& V)
{
    static_assert(!std::is_same_v<itype, DeviceArray<dtype, M>>);
    DeviceArray<dtype, M> U;
    if constexpr (std::is_same_v<dtype, float>)
        U.buf = vector_mul_scalar_float(V.buf, A, M);
    else
        std::cout << "operator * : dtype not implemented!" << std::endl;
    return U;
}

extern "C" float* vector_div_scalar_float(const float* V, float A, size_t N);

template <typename itype, typename dtype, size_t M>
DeviceArray<dtype, M> operator / (DeviceArray<dtype, M> const& V, itype A)
{
    static_assert(!std::is_same_v<itype, DeviceArray<dtype, M>>);
    DeviceArray<dtype, M> U;
    if constexpr (std::is_same_v<dtype, float>)
        U.buf = vector_div_scalar_float(V.buf, A, M);
    else
        std::cout << "operator / : dtype not implemented!" << std::endl;
    return U;
}

template <typename dtype>
inline dtype dot(DeviceVector<dtype> const& V, DeviceVector<dtype> const& U)
{
    assert(V.size == U.size);
    return dot_vector_float(V.buf, U.buf, V.size);
}

template <typename dtype, size_t M>
inline dtype dot(DeviceVector<dtype> const& V, DeviceArray<dtype, M> const& U)
{
    assert(V.size == M);
    if constexpr (std::is_same_v<dtype, float>)
        return dot_vector_float(V.buf, U.buf, M);
    else
        std::cout << "dot: dtype not implemented!" << std::endl;
}

template <typename dtype, size_t M>
inline dtype dot(DeviceArray<dtype, M> const& V, DeviceArray<dtype, M> const& U)
{
    if constexpr (std::is_same_v<dtype, float>)
        return dot_vector_float(V.buf, U.buf, M);
    else
        std::cout << "dot: dtype not implemented!" << std::endl;
}

template <typename dtype, size_t M>
DeviceVector<dtype> dot(DeviceArray<dtype, M> const& T,
                        DeviceVector<DeviceArray<dtype, M>> const& X,
                        dtype bias = dtype())
{
    size_t N = X.size();
    DeviceVector<dtype> Y(N);
    for (size_t i = 0; i < N; ++i)
    {
        Y[i] = dot(X[i], T) + bias;
    }
    return Y;
}