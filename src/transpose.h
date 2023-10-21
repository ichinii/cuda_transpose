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
    static_assert(BLOCK_DIM_Y <= BLOCK_DIM_X);
    static_assert(BLOCK_DIM_X % BLOCK_DIM_Y == 0);

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
    // TODO: select good sqrt function
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
