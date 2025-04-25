#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

#include "vectors_host.hpp"
#include "vectors_device.hpp"

#ifdef LIMIT
    constexpr size_t limit = LIMIT;
#else
    constexpr size_t limit = SIZE_MAX; // default: read all rows
#endif

#ifdef N_FEATURES
    constexpr size_t n_features = N_FEATURES;
#else
    constexpr size_t n_features = 22; // default: value for CTR classification
#endif

#ifdef N_FIELDS
    constexpr size_t n_fields = N_FIELDS;
#else
    constexpr size_t n_fields = 24; // default: value for "train.csv"
#endif


template <typename dtype = float, size_t M = n_features>
class Dataset
{
public:
    std::array<std::string, M> features;
    std::vector<std::array<dtype, M>> X;
    std::vector<dtype> Y;

    Dataset() = default;

    Dataset(std::string const& path, std::string const& target_label,
            std::vector<std::string> ignore_features = std::vector<std::string>(),
            size_t start_row = 0, size_t n_rows = limit)
    {
        X.reserve(n_rows);
        Y.reserve(n_rows);
        read_csv(path, target_label, ignore_features, start_row, n_rows);
    }

    template <typename T>
    static bool contains(std::vector<T> const& container, T item)
    {
        return std::find(container.begin(), container.end(), item) != container.end();
    }

    static dtype cast_data(std::string const& data)
    {
        return static_cast<dtype>(std::stol(data));
    }

    static constexpr auto special_cond = [](size_t i){return i >= 4 && i <= 13;};

    static dtype cast_special_data(std::string const& data)
    {
        return static_cast<dtype>(std::stoul(data, nullptr, 16));
    }

    void read_csv(std::string const& path, std::string const& target_label,
                  std::vector<std::string> ignore_features = std::vector<std::string>(),
                  size_t start_row = 0, size_t n_rows = limit)
    {
        std::ifstream fin(path);
        if (!fin.good())
            throw std::runtime_error("Bad file read");
        std::string line;
        std::string field;
        size_t target_i = SIZE_MAX;
        std::vector<bool> ignore_i(n_fields);
        std::vector<bool> special_i(n_fields);

        getline(fin, line);
        {
        std::stringstream linestream(line);
        size_t dest_i = 0;
        for (size_t i = 0; getline(linestream, field, ','); ++i)
        {
            if (field == target_label)
            {
                target_i = i;
            }
            else if (contains(ignore_features, field))
            {
                ignore_i[i] = true;
            }
            else
            {
                if (special_cond(i))
                {
                    special_i[i] = true;
                }
                features[dest_i] = field;
                ++dest_i;
            }
        }
        if (target_i == SIZE_MAX)
        {
            std::cout << "No target label of name " << target_label << " found!" << std::endl;
        }
        }
        
        std::array<dtype, M> x;
        size_t N = 0;
        for (; getline(fin, line) && N < start_row; ++N);

        for (; getline(fin, line) && N < start_row + n_rows; ++N)
        {
            std::stringstream linestream(line);
            size_t dest_i = 0;
            for (size_t i = 0; getline(linestream, field, ','); ++i)
            {
                if (i == target_i)
                {
                    Y.push_back(cast_data(field));
                }
                else if (ignore_i[i])
                {
                    continue;
                }
                else if (special_i[i])
                {
                    x[dest_i] = cast_special_data(field);
                    ++dest_i;
                }
                else
                {
                    x[dest_i] = cast_data(field);
                    ++dest_i;
                }
            }
            X.push_back(x);
        }
    }

    void print_shape(std::ostream& out = std::cout) const
    {
        out << "X:(" << X.size() << ", " << X[0].size() << "), Y:(" << Y.size() << ")" << std::endl;
    }
};

template <typename dtype = float, size_t M = n_features>
class DeviceDataset
{
public:
    std::array<std::string, M> features;
    DeviceMatrix<dtype, M> X;
    DeviceVector<dtype> Y;

    DeviceDataset() = default;

    DeviceDataset(std::string const& path, std::string const& target_label,
                  std::vector<std::string> ignore_features = std::vector<std::string>(),
                  size_t start_row = 0, size_t n_rows = limit)
        :X(DeviceMatrix<dtype, M>::get_empty(n_rows)),
         Y(DeviceVector<dtype>::get_empty(n_rows))
    {
        read_csv(path, target_label, ignore_features, start_row, n_rows);
    }

    template <typename T>
    static bool contains(std::vector<T> const& container, T item)
    {
        return std::find(container.begin(), container.end(), item) != container.end();
    }

    static dtype cast_data(std::string const& data)
    {
        return static_cast<dtype>(std::stol(data));
    }

    static constexpr auto special_cond = [](size_t i){return i >= 4 && i <= 13;};

    static dtype cast_special_data(std::string const& data)
    {
        return static_cast<dtype>(std::stoul(data, nullptr, 16));
    }

    void read_csv(std::string const& path, std::string const& target_label,
                  std::vector<std::string> ignore_features = std::vector<std::string>(),
                  size_t start_row = 0, size_t n_rows = limit)
    {
        std::ifstream fin(path);
        if (!fin.good())
            throw std::runtime_error("Bad file read");
        std::string line;
        std::string field;
        size_t target_i = SIZE_MAX;
        std::vector<bool> ignore_i(n_fields);
        std::vector<bool> special_i(n_fields);

        getline(fin, line);
        {
        std::stringstream linestream(line);
        size_t dest_i = 0;
        for (size_t i = 0; getline(linestream, field, ','); ++i)
        {
            if (field == target_label)
            {
                target_i = i;
            }
            else if (contains(ignore_features, field))
            {
                ignore_i[i] = true;
            }
            else
            {
                if (special_cond(i))
                {
                    special_i[i] = true;
                }
                features[dest_i] = field;
                ++dest_i;
            }
        }
        if (target_i == SIZE_MAX)
        {
            std::cout << "No target label of name " << target_label << " found!" << std::endl;
        }
        }
        if (n_rows == limit)
            std::cout << "Reading all lines starting from line " << start_row << std::endl;
        else
            std::cout << "Reading " << n_rows << " lines starting from line " << start_row << std::endl;
        // WARN: might get bad_alloc here

        dtype* X_buf = new dtype[n_rows * M];
        dtype* row_ptr = X_buf;
        std::vector<dtype> Y_buf;
        Y_buf.reserve(n_rows);

        size_t N = 0;
        for (; getline(fin, line) && N < start_row; ++N);

        for (; getline(fin, line) && N < start_row + n_rows; ++N)
        {
            std::stringstream linestream(line);
            size_t dest_i = 0;
            for (size_t i = 0; getline(linestream, field, ','); ++i)
            {
                if (i == target_i)
                {
                    Y_buf.push_back(cast_data(field));
                }
                else if (ignore_i[i])
                {
                    continue;
                }
                else if (special_i[i])
                {
                    row_ptr[dest_i] = cast_special_data(field);
                    ++dest_i;
                }
                else
                {
                    row_ptr[dest_i] = cast_data(field);
                    ++dest_i;
                }
            }
            row_ptr += M;
        }
        cudaMemcpy(Y.data(), Y_buf.data(), Y_buf.size() * sizeof(dtype), cudaMemcpyHostToDevice);
        cudaMemcpy(X.data(), X_buf, n_rows * M * sizeof(dtype), cudaMemcpyHostToDevice);
        delete[] X_buf;
    }

    void print_shape(std::ostream& out = std::cout) const
    {
        out << "X:(" << X.size() << ", " << M << "), Y:(" << Y.size() << ")" << std::endl;
    }
};


template <typename dtype = float, size_t M = n_features>
class StandardScaler
{
public:
    std::array<dtype, M> mean;
    std::array<dtype, M> std;
#ifdef USE_CUDA
    void fit(DeviceMatrix<dtype, M> const& X)
    {
        welford<dtype, M>(X.data(), X.size(), mean, std);
    }

    class TransformerFunctor
    {// Device functors have to be placed outside of __host__ functions to be visible.
    public:
        DeviceArray<dtype, M> mean;
        DeviceArray<dtype, M> std;
        dtype* mean_ptr = nullptr;
        dtype* std_ptr = nullptr;

        TransformerFunctor(std::array<dtype, M> const& mean,
                            std::array<dtype, M> const& std)
            :mean(to_device(mean)), std(to_device(std))
        {
            mean_ptr = this->mean.data();
            std_ptr = this->std.data();
        }

        __device__
        void operator () (dtype* row_ptr) const
        {
            for (size_t i = 0; i < M; ++i)
            {
                if (std_ptr[i] != 0)
                    row_ptr[i] = (row_ptr[i] - mean_ptr[i]) / std_ptr[i];
                else
                    row_ptr[i] = 0; // if there's no variance then that feature is invalidated.
            }
        }
    };

    void transform(DeviceMatrix<dtype, M> & X)
    {
        matrix_inplace_transform<dtype>(X.data(), M, X.size(), TransformerFunctor(mean, std));
    }
#else
    void fit(std::vector<std::array<dtype, M>> const& X)
    {
        // Welford's algorithm:
        mean.fill(0.0);
        std::array<dtype, M> m_sq = {0};
        size_t N = 0;

        for (std::array<dtype, M> x: X)
        {
            ++N;
            std::array<dtype, M> delta = x - mean;
            mean += delta / N;
            std::array<dtype, M> delta2 = x - mean;

            for (size_t j = 0; j < M; ++j)
            {// that's a dot
                m_sq[j] += delta[j] * delta2[j];
            }
        }

        for (size_t j = 0; j < M; ++j)
        {
            dtype variance = m_sq[j] / (N - 1);
            std[j] = sqrt(variance);
        }
    }

    void transform(std::vector<std::array<dtype, M>> & X)
    {
        for (size_t i = 0; i < X.size(); ++i)
        {
            std::array<dtype, M> & x = X[i];
            for (size_t j = 0; j < M; ++j)
            {
                if (std[j] != 0)
                    x[j] = (x[j] - mean[j]) / std[j];
                else 
                    x[j] = 0; // if there's no variance then that feature is invalidated.
            }
        }
    }
#endif
};