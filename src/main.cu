#include <iostream>
#include <sstream>
#include <chrono>
#include <tuple>
#include <thread>
#include <vector>
#include <type_traits>

#include <glm/glm.hpp>

// constexpr const unsigned int W = 256;
constexpr const unsigned int W = 1<<10;
constexpr const unsigned int S = W*W;
static_assert(W % 32 == 0);
static_assert(W <= 0xffffffff/2);

constexpr const unsigned int BLOCK_SIZE = 128;
[[maybe_unused]] constexpr const unsigned int GRID_SIZE = S / BLOCK_SIZE;
[[maybe_unused]] constexpr const unsigned int GRID_SIZE_STRIDED = std::min(128u, GRID_SIZE);
static_assert(S == GRID_SIZE * BLOCK_SIZE);

using namespace glm;
using namespace std::chrono_literals;

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

template <typename T>
__global__
void transpose_coalesced_bankconflict(T *dst, T *src, int w) {
    unsigned int o0 = blockIdx.x * 32 + blockIdx.y * 32 * w;
    unsigned int o1 = blockIdx.y * 32 + blockIdx.x * 32 * w;

    constexpr const uint8 X = 32;
    __shared__ int data[32*X];

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.y + y + threadIdx.x * X;

        T p = src[pi + o0];
        data[di] = p;
    }

    __syncthreads();

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.x + (y + threadIdx.y) * X;

        dst[pi + o1] = data[di];
    }
}

template <typename T>
__global__
void transpose_coalesced(T *dst, T *src, int w) {
    unsigned int o0 = blockIdx.x * 32 + blockIdx.y * 32 * w;
    unsigned int o1 = blockIdx.y * 32 + blockIdx.x * 32 * w;

    constexpr const uint8 X = 33;
    __shared__ int data[32*X];

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.y + y + threadIdx.x * X;

        T p = src[pi + o0];
        data[di] = p;
    }

    __syncthreads();

    for (int y = 0; y < 32; y += blockDim.y) {
        int pi = threadIdx.x + (y + threadIdx.y) * w;
        int di = threadIdx.x + (y + threadIdx.y) * X;

        dst[pi + o1] = data[di];
    }
}

template <typename T>
__global__
void transposeCoalesced(T *odata, const T *idata) {
    constexpr const int TILE_DIM = 32;
    constexpr const int BLOCK_ROWS = 4;

    __shared__ float tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

#define CHECK_LAST_CUDA_ERROR() checkLastError(__FILE__, __LINE__)
inline void checkLastError(const char* const file, const int line) {
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl
            << cudaGetErrorString(err) << std::endl;
    }
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
using ReduceOp = T(*)(T, T);

template <typename T>
__device__
T op_add(T a, T b) {
    return a + b;
}

__device__
uint8 op_binary_or(uint8 a, uint8 b) {
    return a && b;
}

__device__ ReduceOp<int32> p_op_add_int32 = op_add<int32>;
__device__ ReduceOp<uint8> p_op_binary_or = op_binary_or;

template <typename T>
ReduceOp<T> get_reduce_op_add() {
    ReduceOp<T> op;
    static_assert(std::is_same_v<T, int32>);
    cudaMemcpyFromSymbol(&op, p_op_add_int32, sizeof(ReduceOp<T>));
    return op;
}

ReduceOp<uint8> get_reduce_op_binary_and() {
    ReduceOp<uint8> op;
    cudaMemcpyFromSymbol(&op, p_op_binary_or, sizeof(ReduceOp<uint8>));
    return op;
}

constexpr const uint8 identity_binary_op = 1;
constexpr const int32 identity_op_add = 0;

/* begin
 * Mark Harris NVIDIA Developer Technology
 */
template <unsigned int blockSize, typename T>
__device__
void warpReduce(volatile T *sdata, unsigned int tid, ReduceOp<T> op) {
    if (blockSize >= 64) sdata[tid] = op(sdata[tid], sdata[tid + 32]);
    if (blockSize >= 32) sdata[tid] = op(sdata[tid], sdata[tid + 16]);
    if (blockSize >= 16) sdata[tid] = op(sdata[tid], sdata[tid + 8]);
    if (blockSize >= 8)  sdata[tid] = op(sdata[tid], sdata[tid + 4]);
    if (blockSize >= 4)  sdata[tid] = op(sdata[tid], sdata[tid + 2]);
    if (blockSize >= 2)  sdata[tid] = op(sdata[tid], sdata[tid + 1]);
}

template <unsigned int blockSize, typename T>
__global__
void reduce_k(T *g_odata, T *g_idata, unsigned int n, ReduceOp<T> op, T identity_op) {
    extern __shared__ /*__align__(sizeof(T))*/ uint8 sdata_[];
    T *sdata = reinterpret_cast<T*>(sdata_);
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = identity_op;
    while (i < n) { sdata[tid] = op(sdata[tid], op(g_idata[i], g_idata[i+blockSize])); i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid] = op(sdata[tid], sdata[tid + 64]);  } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid, op);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
/* end
 * Mark Harris NVIDIA Developer Technology
 */

constexpr inline bool is_power_of_2(int n) {
    return (n & (n-1)) == 0;
}

template <typename T>
void reduce_once(T *dst, T *src, int b, int t, int n, ReduceOp<T> op, T identity_op) {
    assert(t <= 512);
    assert(is_power_of_2(t));

#define BlockSizeCase(t) case t: reduce_k<t, T><<<b, t, t*sizeof(T)>>>(dst, src, n, op, identity_op); break;
    switch (t) {
        BlockSizeCase(512)
        BlockSizeCase(256)
        BlockSizeCase(128)
        BlockSizeCase( 64)
        BlockSizeCase( 32)
        BlockSizeCase( 16)
        BlockSizeCase(  8)
        BlockSizeCase(  4)
        BlockSizeCase(  2)
        BlockSizeCase(  1)
        default: assert(false); break;
    }
#undef sum_case
}

template <typename T>
void dump(T* data_d, uint n, uint w=W) {
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

template <typename T>
T reduce(T *src, ReduceOp<T> op, T identity_op) {
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

    bool result_h = reduce(result, get_reduce_op_binary_and(), identity_binary_op);
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
        << experiment<uint64>(expected, c, fn);
    return ss.str();
}

using InitRangePred = int8(*)(uint);

__device__
int8 pred_one(uint i) {
    return 1;
}

__device__
InitRangePred p_pred_one = pred_one;

template <typename T>
__global__
void init_range(T *dst, InitRangePred pred) {
    uint i = threadIdx.x + blockIdx.x*blockDim.x;
    dst[i] = pred(i);
}

InitRangePred get_pred_one() {
    InitRangePred pred;
    cudaMemcpyFromSymbol(&pred, p_pred_one, sizeof(InitRangePred));
    return pred;
}

int main()
{
    std::this_thread::sleep_for(50ms);
    std::stringstream ss;
    ss << std::boolalpha;

    uint8 *expected;
    cudaMalloc(&expected, S * sizeof(*expected));
    init_data<true>(expected);

    {
        ss << "\tlets first test the helper functions" << std::endl;
        {
            int32 *data;
            cudaMalloc(&data, S * sizeof(*data));
            init_range<<<GRID_SIZE, BLOCK_SIZE>>>(data, get_pred_one());
            auto sum = reduce(data, get_reduce_op_add<int32>(), identity_op_add);
            cudaFree(data);

            ss
                << "reduce(op=add)       ok=" << (sum == S) << std::endl;
        }
        {
            uint8 *data;
            cudaMalloc(&data, S * sizeof(*data));
            init_range<<<GRID_SIZE, BLOCK_SIZE>>>(data, get_pred_one());
            auto result_true = reduce(data, get_reduce_op_binary_and(), identity_binary_op);

            init_range<<<GRID_SIZE, BLOCK_SIZE>>>(data, get_pred_one());
            uint8 zero = 0;
            cudaMemcpy(data, &zero, sizeof(uint8), cudaMemcpyHostToDevice);
            auto result_false = reduce(data, get_reduce_op_binary_and(), identity_binary_op);
            cudaFree(data);

            ss
                << "reduce(op=binary_or) ok=" << static_cast<bool>(result_true) << std::endl
                << "reduce(op=binary_or) ok=" << !static_cast<bool>(result_false) << std::endl;
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
                << "compare(equal)       ok=" << static_cast<bool>(result_true) << std::endl
                << "compare(different)   ok=" << !static_cast<bool>(result_false) << std::endl
                << "compare(different)   ok=" << !static_cast<bool>(result_false2) << std::endl;
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
        transpose_coalesced_bankconflict<<<{W/32u, W/32u, 1u}, {c.x, c.y, 1u}>>>(output, input, W);
    };

    [[maybe_unused]] auto fn_2 = [] <typename T> (uvec2 c, T *output, T *input) {
        transpose_coalesced<<<{W/32u, W/32u, 1u}, {c.x, c.y, 1u}>>>(output, input, W);
    };

    [[maybe_unused]] auto fn_3 = [] <typename T> (uvec2 c, T *output, T *input) {
        transposeCoalesced<<<{W/32u, W/32u, 1u}, {c.x, c.y, 1u}>>>(output, input);
    };

    ss
        << "\ttranspose_naive" << std::endl
        << run_experiments_for_element_sizes(expected, {4, 32}, fn_0)
        << std::endl

        << "\ttranspose_coalesced_bankconflict" << std::endl
        << run_experiments_for_element_sizes(expected, {32, 4}, fn_1)
        << std::endl

        << "\ttranspose_coalesced" << std::endl
        << run_experiments_for_element_sizes(expected, {32, 4}, fn_2)
        << std::endl

        << "\ttransposeCoalesced" << std::endl
        << run_experiments_for_element_sizes(expected, {32, 4}, fn_3)
        << std::endl
    ;

    cudaFree(expected);
    std::cout << ss.str();

    return 0;
}