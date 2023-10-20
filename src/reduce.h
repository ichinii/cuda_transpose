#pragma once

template <typename T>
struct AddOp {
    __device__
    __forceinline__
    T operator()(T a, T b) { return a + b; }
};

template <typename T>
struct BinaryAndOp {
    __device__
    __forceinline__
    T operator()(T a, T b) { return a && b; }
};

template <typename T, typename OpFn>
extern T reduce(T *src, OpFn op, T identity_op);
