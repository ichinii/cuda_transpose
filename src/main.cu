#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <thread>
#include <vector>
#include <type_traits>

#include <glm/glm.hpp>

using namespace glm;
using namespace std::chrono_literals;

// constexpr const unsigned int W = 128;
constexpr const uint W = 1<<12;
constexpr const uint S = W*W;
static_assert(W % 32 == 0);
static_assert(W <= 1<<16);

constexpr const unsigned int BLOCK_SIZE = 128;
[[maybe_unused]] constexpr const unsigned int GRID_SIZE = S / BLOCK_SIZE;
[[maybe_unused]] constexpr const unsigned int GRID_SIZE_STRIDED = std::min(128u, GRID_SIZE);
static_assert(S == GRID_SIZE * BLOCK_SIZE);

__device__
__forceinline__
int coord_to_id(ivec2 coord, int w) {
    return coord.x + coord.y*w;
}

__device__
__forceinline__
ivec2 id_to_coord(int id, int w) {
    return ivec2(id % w, id / w);
}

template <typename T>
__global__
void transpose_naive(T *dst, T *src, int w) {
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int dst_id = coord_to_id(ivec2(x, y), w);
    int src_id = coord_to_id(ivec2(y, x), w);
    dst[dst_id] = src[src_id];
}

template <uint BLOCK_DIM_X, uint BLOCK_DIM_Y, typename T>
__global__
void transpose_coalesced_bankconflict(T *dst, T *src) {
    __shared__ T data[BLOCK_DIM_X][BLOCK_DIM_X];

    uint w = gridDim.x * BLOCK_DIM_X;
    uint ox = threadIdx.x + blockIdx.x*BLOCK_DIM_X;
    uint oy = threadIdx.y + blockIdx.y*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        data[threadIdx.y+y][threadIdx.x] = src[ox + (oy+y) * w];

    __syncthreads();
    ox = threadIdx.x + blockIdx.y*BLOCK_DIM_X;
    oy = threadIdx.y + blockIdx.x*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        dst[ox + (oy+y) * w] = data[threadIdx.x][threadIdx.y+y];
}

template <uint BLOCK_DIM_X, uint BLOCK_DIM_Y, typename T>
__global__
void transpose_coalesced(T *dst, T *src) {
    __shared__ T data[BLOCK_DIM_X][BLOCK_DIM_X + 1];

    uint w = gridDim.x * BLOCK_DIM_X;
    uint ox = threadIdx.x + blockIdx.x*BLOCK_DIM_X;
    uint oy = threadIdx.y + blockIdx.y*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        data[threadIdx.y+y][threadIdx.x] = src[ox + (oy+y) * w];

    __syncthreads();
    ox = threadIdx.x + blockIdx.y*BLOCK_DIM_X;
    oy = threadIdx.y + blockIdx.x*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        dst[ox + (oy+y) * w] = data[threadIdx.x][threadIdx.y+y];
}

__device__
float Q_rsqrt(float number) {
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;    // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck? 
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    // y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration,

    return y;
}

__device__
ivec2 index_to_tricoord(int i) {
    // int y = 1.0/Q_rsqrt(0.25f + 2.0f*i) - 0.5f;
    int y = sqrt(0.25f + 2.0f*i) - 0.5f;
    int x = i - y*(y+1)/2;
    return ivec2(x, y);
}

int tricoord_to_index(ivec2 v) {
    return v.x + v.y*(v.y+1)/2;
}

template <uint BLOCK_DIM_X, uint BLOCK_DIM_Y, typename T>
__global__
void transpose_triangle(T *src, uint w) {
    uvec2 tile = index_to_tricoord(blockIdx.x);

    __shared__ T data0[BLOCK_DIM_X][BLOCK_DIM_X + 1];
    __shared__ T data1[BLOCK_DIM_X][BLOCK_DIM_X + 1];

    uint ox0 = threadIdx.x + tile.x*BLOCK_DIM_X;
    uint oy0 = threadIdx.y + tile.y*BLOCK_DIM_X;
    uint ox1 = threadIdx.x + tile.y*BLOCK_DIM_X;
    uint oy1 = threadIdx.y + tile.x*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y) {
        data0[threadIdx.y+y][threadIdx.x] = src[ox0 + (oy0+y) * w];
        data1[threadIdx.y+y][threadIdx.x] = src[ox1 + (oy1+y) * w];
    }

    __syncthreads();

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y) {
        src[ox0 + (oy0+y) * w] = data1[threadIdx.x][threadIdx.y+y];
        src[ox1 + (oy1+y) * w] = data0[threadIdx.x][threadIdx.y+y];
    }
}

template <uint BLOCK_DIM_X, uint BLOCK_DIM_Y, typename T>
__global__
void transpose_triangle_diag(T *src) {
    __shared__ T data[BLOCK_DIM_X][BLOCK_DIM_X + 1];

    uint w = gridDim.x * BLOCK_DIM_X;
    uint ox = threadIdx.x + blockIdx.x*BLOCK_DIM_X;
    uint oy = threadIdx.y + blockIdx.x*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        data[threadIdx.y+y][threadIdx.x] = src[ox + (oy+y) * w];

    __syncthreads();

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y)
        src[ox + (oy+y) * w] = data[threadIdx.x][threadIdx.y+y];
}

template <uint BLOCK_DIM_X, uint BLOCK_DIM_Y, typename T>
__global__
void transpose_triangle_internal(T *src, uint w) {
    uvec2 tile = index_to_tricoord(blockIdx.x);
    tile.y += 1;

    __shared__ T data0[BLOCK_DIM_X][BLOCK_DIM_X + 1];
    __shared__ T data1[BLOCK_DIM_X][BLOCK_DIM_X + 1];

    uint ox0 = threadIdx.x + tile.x*BLOCK_DIM_X;
    uint oy0 = threadIdx.y + tile.y*BLOCK_DIM_X;
    uint ox1 = threadIdx.x + tile.y*BLOCK_DIM_X;
    uint oy1 = threadIdx.y + tile.x*BLOCK_DIM_X;

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y) {
        data0[threadIdx.y+y][threadIdx.x] = src[ox0 + (oy0+y) * w];
        data1[threadIdx.y+y][threadIdx.x] = src[ox1 + (oy1+y) * w];
    }

    __syncthreads();

    for (uint y = 0; y < BLOCK_DIM_X; y += BLOCK_DIM_Y) {
        src[ox0 + (oy0+y) * w] = data1[threadIdx.x][threadIdx.y+y];
        src[ox1 + (oy1+y) * w] = data0[threadIdx.x][threadIdx.y+y];
    }
}

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

template <typename T>
struct AdditionOp {
    __device__ __host__
    __forceinline__
    T operator()(T a, T b) { return a + b; }
};

template <typename T>
struct BinaryAndOp {
    __device__ __host__
    __forceinline__
    T operator()(T a, T b) { return a && b; }
};

constexpr inline bool is_power_of_2(int n) {
    return (n & (n-1)) == 0;
}

template <unsigned int blockSize, typename T, typename OpFn>
__device__
__forceinline__
T reduce_warp(T x, OpFn op) {
#pragma unroll // TODO: find a way to make sure this loop is unrolled
    for (int i = std::min(32u, blockSize); 2 <= i; i /= 2)
        x = op(x, __shfl_down_sync(0xffffffff, x, i/2));
    return x;
}

template <unsigned int blockSize, typename T, typename OpFn>
__global__
void reduce_k(T *g_odata, T *g_idata, unsigned int n, OpFn op, T identity_op) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T x = identity_op;
    while (i < n) { x = op(x, op(g_idata[i], g_idata[i+blockSize])); i += gridSize; }
    __syncthreads();

    x = reduce_warp<blockSize>(x, op);

    if constexpr (blockSize >= 64) {
        __shared__ T sdata[blockSize / 32];

        if (tid % 32 == 0) {
            sdata[tid / 32] = x;
        }

        __syncthreads();

        if (tid < blockSize / 32) {
            x = sdata[tid];
            x = reduce_warp<blockSize / 32>(x, op);
        }
    }

    if (tid == 0)
        g_odata[blockIdx.x] = x;
}

template <typename T, typename OpFn>
void reduce_once(T *dst, T *src, int b, int t, int n, OpFn op, T identity_op) {
    assert(t <= 1024);
    assert(is_power_of_2(t));
    assert(n % (t*2) == 0);

#define BlockSizeCase(t) case t: reduce_k<t, T><<<b, t>>>(dst, src, n, op, identity_op); break;
    switch (t) {
        BlockSizeCase(1024)
        BlockSizeCase( 512)
        BlockSizeCase( 256)
        BlockSizeCase( 128)
        BlockSizeCase(  64)
        BlockSizeCase(  32)
        BlockSizeCase(  16)
        BlockSizeCase(   8)
        BlockSizeCase(   4)
        BlockSizeCase(   2)
        BlockSizeCase(   1)
        default: assert(false); break;
    }
#undef BlockSizeCase
}

// TODO: T in OpFn can be omitted on caller site
// ^ something like template <typename> typename OpFn> and use parameter OpFn<T>
// TODO: op should not be a parameter, bc its an empty class anyways
// TODO: for OpFn enable simply passing labmda.
// ^ to reduce instances of this template, are calls wtih identical lambdas collapsed into single instance?
template <typename T, typename OpFn>
T reduce(T *src, OpFn op, T identity_op) {
    T *dst;
    cudaMalloc(&dst, S * sizeof(T));
    T *markForFree = dst;

    uint r = BLOCK_SIZE*2;
    uint n = S;

    while (1 < n) {
        uint b = std::max(1u, n/r);
        b = std::min(b, GRID_SIZE_STRIDED);
        uint t = std::min(BLOCK_SIZE, n/2);
        reduce_once(dst, src, b, t, n, op, identity_op);
        n = b;
        std::swap(dst, src);
    }
    std::swap(dst, src);

    T result;
    cudaMemcpy(&result, dst, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(markForFree);
    return result;
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
void init_range_with_one(T *dst) {
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    dst[i] = 1;
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
            init_range_with_one<<<GRID_SIZE, BLOCK_SIZE>>>(data);
            int32 sum = reduce(data, AdditionOp<int32>(), 0);
            cudaFree(data);

            ss
                << "reduce(op=add)        ok=" << (sum == S) << std::endl;
        }
        {
            uint8 *data;
            cudaMalloc(&data, S * sizeof(*data));
            init_range_with_one<<<GRID_SIZE, BLOCK_SIZE>>>(data);
            auto result_true = reduce(data, BinaryAndOp<uint8>(), static_cast<uint8>(1));

            init_range_with_one<<<GRID_SIZE, BLOCK_SIZE>>>(data);
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
    std::cout << ss.str();

    return 0;
}
