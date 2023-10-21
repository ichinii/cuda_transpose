#include "common.h"
#include "reduce.h"

constexpr inline bool is_power_of_2(int n) {
    return (n & (n-1)) == 0;
}

template <uint warpSize, typename T, typename OpFn>
__device__
__forceinline__
T reduce_warp(uint tid, T x, OpFn op) {
#pragma unroll // TODO: find a way to make sure this loop is unrolled
    for (uint i = warpSize / 2; 1 <= i; i >>= 1) {
        uint active = __ballot_sync(0xffffffff, tid < i);
        x = op(x, __shfl_down_sync(active, x, i));
    }
    return x;
}

template <uint blockSize, typename T, typename OpFn>
__global__
void reduce_k(T *dst, T *src, uint n, OpFn op, T identity_op) {
    uint tid = threadIdx.x;
    uint i = blockIdx.x*(blockSize*2) + tid;
    uint gridSize = blockSize*2*gridDim.x;

    T x = identity_op;
    while (i < n) {
        x = op(x, op(src[i], src[i+blockSize])); i += gridSize;
    }

    x = reduce_warp<std::min(32u, blockSize)>(tid, x, op);

    if constexpr (blockSize >= 64) {
        __shared__ T sdata[32];

        if (tid % 32 == 0) {
            sdata[tid / 32] = x;
        }

        __syncthreads();

        if (tid < 32) {
            x = sdata[tid];
            x = reduce_warp<blockSize / 32>(tid, x, op);
        }
    }

    if (tid == 0)
        dst[blockIdx.x] = x;
}

template <typename T, typename OpFn>
void reduce_once(T *dst, T *src, int b, int t, int n, OpFn op, T identity_op) {
    assert(t <= 1024 && is_power_of_2(t));
    assert(t*2 <= n && n % (t*2) == 0);

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
    static_assert(1 < BLOCK);
    constexpr uint r = BLOCK*2;

    T *dst;
    cudaMalloc(&dst, S * sizeof(T) / r);
    T *markForFree = dst;

    uint n = S;
    while (1 < n) {
        uint b = clamp(n/r, 1u, GRID_STRIDED);
        uint t = min(BLOCK, n/2);
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

template
uint8 reduce<uint8, BinaryAndOp<uint8>>(uint8 *src, BinaryAndOp<uint8> op, uint8 identity_op);

template
int32 reduce<int32, AddOp<int32>>(int32 *src, AddOp<int32> op, int32 identity_op);
