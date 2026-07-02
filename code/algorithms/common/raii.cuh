// common/raii.cuh
// 재사용 코어: CUDA 런타임 RAII 유틸리티 (알고리즘 무관).
#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// 디바이스 메모리 RAII. 복사 금지, 이동 허용.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) : count_(count) {
        CHECK_CUDA(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }
    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), count_(o.count_) {
        o.ptr_ = nullptr; o.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) { reset(); ptr_ = o.ptr_; count_ = o.count_; o.ptr_ = nullptr; o.count_ = 0; }
        return *this;
    }

    T*       data()        { return ptr_; }
    const T* data()  const { return ptr_; }
    size_t   size()  const { return count_; }
    size_t   bytes() const { return count_ * sizeof(T); }

    void copyFromHost(const T* h) { CHECK_CUDA(cudaMemcpy(ptr_, h, bytes(), cudaMemcpyHostToDevice)); }
    void copyToHost(T* h)   const { CHECK_CUDA(cudaMemcpy(h, ptr_, bytes(), cudaMemcpyDeviceToHost)); }

private:
    void reset() { if (ptr_) { cudaFree(ptr_); ptr_ = nullptr; } count_ = 0; }
    T*     ptr_   = nullptr;
    size_t count_ = 0;
};

// cudaEvent 기반 타이머 RAII. start() → 커널 → stop() 이 경과 ms 반환.
class GpuTimer {
public:
    GpuTimer()  { CHECK_CUDA(cudaEventCreate(&start_)); CHECK_CUDA(cudaEventCreate(&stop_)); }
    ~GpuTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    void start(cudaStream_t s = 0) { CHECK_CUDA(cudaEventRecord(start_, s)); }
    float stop(cudaStream_t s = 0) {
        CHECK_CUDA(cudaEventRecord(stop_, s));
        CHECK_CUDA(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
private:
    cudaEvent_t start_{}, stop_{};
};
