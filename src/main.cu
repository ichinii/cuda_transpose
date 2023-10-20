#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <thread>
#include <vector>
#include <type_traits>

#include "common.h"
#include "reduce.h"
#include "transpose.h"

using namespace std::chrono_literals;

#define CHECK_LAST_CUDA_ERROR() checkLastError(__FILE__, __LINE__)
inline void checkLastError(const char* const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl
            << cudaGetErrorString(err) << std::endl;
    }
}

template <typename T>
void dump(T* data_d, uint n=S, uint w=W) {
    auto data = std::vector<T>(n);
    cudaMemcpy(data.data(), data_d, n * sizeof(T), cudaMemcpyDeviceToHost);
    std::cout << "\tdump" << std::endl;
    for (int i = 0; i < n; ++i) {
        if (i != 0 && i % w == 0)
            std::cout << std::endl;
        std::cout << (int64)data[i] << " ";
    }
    std::cout << std::endl;
}


template <bool Transposed, typename T>
__global__
void init_data_k(T *output) {
    for (int b = 0; b < blockDim.x; b += blockDim.y) {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.x + b;
        int i = x + y * W;
        if constexpr (Transposed)
            output[i] = x % 256;
        else
            output[i] = y % 256;
    }
}

template <bool Transposed, typename T>
void init_data(T *output) {
    init_data_k<Transposed><<<{W/32u, W/32u, 1u}, {32u, 4u, 1u}>>>(output);
}

template <typename T, typename U>
__global__
void compare_k(uint8 *dst, T *a, U *b) {
    for (int s = 0; s < blockDim.x; s += blockDim.y) {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.x + s;
        int i = x + y * W;
        dst[i] = a[i] == b[i];
    }
}

template <typename T, typename U>
bool compare(T *a, U *b) {
    uint8 *result;
    cudaMalloc(&result, S * sizeof(*result));
    compare_k<<<{W/32u, W/32u, 1u}, {32u, 4u, 1u}>>>(result, a, b);

    bool result_h = reduce(result, BinaryAndOp<uint8>(), static_cast<uint8>(1));
    cudaFree(result);
    return result_h;
}

template <typename Fn>
auto perf(Fn fn) {
    using namespace std::chrono;

    float dt_gpu;
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    cudaDeviceSynchronize();
    auto start = steady_clock::now();

    cudaEventRecord(start_event);
    fn();
    cudaEventRecord(end_event);

    cudaEventSynchronize(end_event);
    auto end = steady_clock::now();

    cudaDeviceSynchronize();
    cudaEventElapsedTime(&dt_gpu, start_event, end_event);
    cudaEventDestroy(end_event);
    cudaEventDestroy(start_event);

    float dt_cpu = duration_cast<microseconds>(end - start).count() / 1000.0f;
    return std::make_tuple(dt_cpu, dt_gpu);
}

template <typename T, typename Fn>
std::string experiment(uint8 *expected, uvec2 c, Fn fn) {
    T *input, *output;
    cudaMalloc(&input, S * sizeof(*input));
    cudaMalloc(&output, S * sizeof(*output));
    init_data<false>(input);

    auto [dt_cpu, dt_gpu] = perf([&] {
        fn(c, output, input);
    });
    CHECK_LAST_CUDA_ERROR();

    auto ok = compare(expected, output);

    cudaFree(input);
    cudaFree(output);

    std::stringstream ss;
    ss
        << "b=" << sizeof(T)
        << "\tdt_cpu=" << dt_cpu
        << "\tdt_gpu=" << dt_gpu
        << std::boolalpha << "\t ok=" << ok
        << std::endl;
    return ss.str();
}

template <typename Fn>
std::string run_experiments_for_element_sizes(uint8 *expected, uvec2 c, Fn fn) {
    std::stringstream ss;
    ss
        << experiment<uint8> (expected, c, fn)
        << experiment<uint16>(expected, c, fn)
        << experiment<uint32>(expected, c, fn)
        << experiment<uint64>(expected, c, fn)
    ;
    return ss.str();
}

template <typename T>
__global__
void set_range(T *dst, T v) {
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    dst[i] = v;
}

template <typename T, typename Fn>
__global__
void set_range_pred(T *dst, Fn pred) {
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    dst[i] = pred(i);
}

int main()
{
    std::this_thread::sleep_for(50ms);
    std::stringstream ss;
    ss << std::boolalpha;

    uint8 *expected;
    cudaMalloc(&expected, S * sizeof(*expected));
    init_data<true>(expected);

    if constexpr (true) {
        ss << "lets first test the helper functions" << std::endl;
        {
            int32 *data;
            cudaMalloc(&data, S * sizeof(*data));
            set_range<<<GRID, BLOCK>>>(data, static_cast<int32>(1));
            int32 sum = reduce(data, AddOp<int32>(), 0);
            cudaFree(data);

            ss
                << "reduce(op=add)        ok=" << (sum == S) << std::endl;
        }
        {
            uint8 *data;
            cudaMalloc(&data, S * sizeof(*data));
            set_range<<<GRID, BLOCK>>>(data, static_cast<uint8>(1));
            auto result_true = reduce(data, BinaryAndOp<uint8>(), static_cast<uint8>(1));

            set_range<<<GRID, BLOCK>>>(data, static_cast<uint8>(1));
            uint8 zero = 0;
            cudaMemcpy(data, &zero, sizeof(uint8), cudaMemcpyHostToDevice);
            auto result_false = reduce(data, BinaryAndOp<uint8>(), static_cast<uint8>(1));
            cudaFree(data);

            ss
                << "reduce(op=binary_and) ok=" << static_cast<bool>(result_true) << std::endl
                << "reduce(op=binary_and) ok=" << !static_cast<bool>(result_false) << std::endl;
        }
        {
            uint8 *data, *data2, *data_transposed;
            cudaMalloc(&data, S * sizeof(*data));
            cudaMalloc(&data2, S * sizeof(*data2));
            cudaMalloc(&data_transposed, S * sizeof(*data_transposed));
            init_data<false>(data);
            init_data<false>(data2);
            init_data<true>(data_transposed);
            uint8 one = 1;
            cudaMemcpy(data2, &one, sizeof(uint8), cudaMemcpyHostToDevice);
            auto result_true = compare(data, data);
            auto result_false = compare(data, data_transposed);
            auto result_false2 = compare(data, data2);
            cudaFree(data);
            cudaFree(data2);
            cudaFree(data_transposed);

            ss
                << "compare(equal)        ok=" << static_cast<bool>(result_true) << std::endl
                << "compare(different)    ok=" << !static_cast<bool>(result_false) << std::endl
                << "compare(different)    ok=" << !static_cast<bool>(result_false2) << std::endl;
        }
        ss << std::endl;
    }

    ss
        << "width: " << W << std::endl
        << "pixels: " << S << std::endl
        << "dt (delta time) in milliseconds" << std::endl
        << std::endl;

    [[maybe_unused]] auto fn_0 = [] <typename T> (uvec2 c, T *output, T *input) {
        transpose_naive<<<{W/c.x, W/c.y, 1u}, {c.x, c.y, 1u}>>>(output, input, W);
    };

    [[maybe_unused]] auto fn_1 = [] <typename T> (uvec2 c, T *output, T *input) {
        constexpr const uint X = 32;
        constexpr const uint Y = 4;
        transpose_coalesced_bankconflict<X, Y><<<{W/X, W/X, 1u}, {X, Y, 1u}>>>(output, input);
    };

    [[maybe_unused]] auto fn_2 = [] <typename T> (uvec2 c, T *output, T *input) {
        constexpr const uint X = 32;
        constexpr const uint Y = 4;
        transpose_coalesced<X, Y><<<{W/X, W/X, 1u}, {X, Y, 1u}>>>(output, input);
    };

    [[maybe_unused]] auto fn_3 = [] <typename T> (uvec2 c, T *&output, T *&input) {
        constexpr const uint X = 32;
        constexpr const uint Y = 4;
        uint B = tricoord_to_index(ivec2(W/X - 1, W/X - 1)) + 1;
        transpose_triangle<X, Y><<<{B, 1u, 1u}, {X, Y, 1u}>>>(input, W);
        std::swap(output, input);
    };

    [[maybe_unused]] auto fn_4 = [] <typename T> (uvec2 c, T *&output, T *&input) {
        constexpr const uint X = 32;
        constexpr const uint Y = 4;
        uint B = tricoord_to_index(ivec2(W/X - 2, W/X - 2)) + 1;
        transpose_triangle_internal<X, Y><<<{B, 1u, 1u}, {X, Y, 1u}>>>(input, W);
        transpose_triangle_diag<X, Y><<<{W/X, 1u, 1u}, {X, Y, 1u}>>>(input);
        std::swap(output, input);
    };

    ss
        << "transpose_naive" << std::endl
        << run_experiments_for_element_sizes(expected, {4, 32}, fn_0)
        << std::endl

        << "transpose_coalesced_bankconflict" << std::endl
        << run_experiments_for_element_sizes(expected, {}, fn_1)
        << std::endl

        << "transpose_coalesced" << std::endl
        << run_experiments_for_element_sizes(expected, {}, fn_2)
        << std::endl

        << "transpose_triangle" << std::endl
        << run_experiments_for_element_sizes(expected, {}, fn_3)
        << std::endl

        << "transpose_triangle separate diag&internal" << std::endl
        << run_experiments_for_element_sizes(expected, {}, fn_4)
        << std::endl
    ;

    cudaFree(expected);
    CHECK_LAST_CUDA_ERROR();

    std::cout << ss.str();

    return 0;
}
