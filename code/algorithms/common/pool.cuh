// common/pool.cuh
// 재사용 코어: 메모리 풀 — 같은 크기 블록 재사용으로 할당/해제 반복 비용 제거.
//   PinnedMemoryPool : 호스트 pinned(page-locked), 빠른 async 전송 스테이징.
//   DeviceMemoryPool : 디바이스. 대안: CUDA 11.2+ cudaMallocAsync/cudaMemPool_t.
#pragma once

#include "raii.cuh"   // CHECK_CUDA

#include <cstddef>
#include <unordered_map>
#include <vector>

struct DeviceAlloc {
    static void* alloc(size_t b) { void* p = nullptr; CHECK_CUDA(cudaMalloc(&p, b)); return p; }
    static void  free(void* p)   { cudaFree(p); }
};
struct PinnedAlloc {
    static void* alloc(size_t b) { void* p = nullptr; CHECK_CUDA(cudaMallocHost(&p, b)); return p; }
    static void  free(void* p)   { cudaFreeHost(p); }
};

template <typename Alloc>
class MemoryPool {
public:
    static MemoryPool& instance() { static MemoryPool pool; return pool; }

    void* acquire(size_t bytes) {
        auto it = free_.find(bytes);
        if (it != free_.end() && !it->second.empty()) {
            void* p = it->second.back(); it->second.pop_back(); return p;
        }
        return Alloc::alloc(bytes);
    }
    void release(void* p, size_t bytes) { if (p) free_[bytes].push_back(p); }
    void clear() {
        for (auto& kv : free_) for (void* p : kv.second) Alloc::free(p);
        free_.clear();
    }
    ~MemoryPool() { clear(); }
    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;
private:
    MemoryPool() = default;
    std::unordered_map<size_t, std::vector<void*>> free_;
};

using PinnedMemoryPool = MemoryPool<PinnedAlloc>;
using DeviceMemoryPool = MemoryPool<DeviceAlloc>;

// 풀에서 빌린 호스트 pinned 버퍼 RAII (소멸 시 풀로 반환).
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(size_t count)
        : count_(count), ptr_(static_cast<T*>(PinnedMemoryPool::instance().acquire(count * sizeof(T)))) {}
    ~PinnedBuffer() { PinnedMemoryPool::instance().release(ptr_, count_ * sizeof(T)); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T*       data()             { return ptr_; }
    const T* data()       const { return ptr_; }
    T&       operator[](size_t i)       { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }
    size_t   size()  const { return count_; }
    size_t   bytes() const { return count_ * sizeof(T); }
private:
    size_t count_;
    T*     ptr_;
};

// 풀에서 빌린 디바이스 버퍼 RAII (반복 할당/해제 경로용).
template <typename T>
class PooledDeviceBuffer {
public:
    explicit PooledDeviceBuffer(size_t count)
        : count_(count), ptr_(static_cast<T*>(DeviceMemoryPool::instance().acquire(count * sizeof(T)))) {}
    ~PooledDeviceBuffer() { DeviceMemoryPool::instance().release(ptr_, count_ * sizeof(T)); }
    PooledDeviceBuffer(const PooledDeviceBuffer&) = delete;
    PooledDeviceBuffer& operator=(const PooledDeviceBuffer&) = delete;

    T*       data()        { return ptr_; }
    const T* data()  const { return ptr_; }
    size_t   size()  const { return count_; }
    size_t   bytes() const { return count_ * sizeof(T); }
    void copyFromHost(const T* h) { CHECK_CUDA(cudaMemcpy(ptr_, h, bytes(), cudaMemcpyHostToDevice)); }
    void copyToHost(T* h)   const { CHECK_CUDA(cudaMemcpy(h, ptr_, bytes(), cudaMemcpyDeviceToHost)); }
private:
    size_t count_;
    T*     ptr_;
};
