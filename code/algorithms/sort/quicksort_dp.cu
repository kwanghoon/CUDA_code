// quicksort_dp.cu
// 슬라이드: part5/chapter36 (Sorting) — CUDA Dynamic Parallelism 데모.
// 부모 커널이 파티션 후 두 자식 커널을 재귀 런치(divide-and-conquer).
// 빌드: -rdc=true -lcudadevrt (CMake: CUDA_SEPARABLE_COMPILATION + CUDA::cudadevrt), SM 3.5+.
#include "../common/raii.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

__device__ __forceinline__ void dswap(int& a, int& b) { int t = a; a = b; b = t; }

__global__ void quicksortDP(int* data, int lo, int hi, int depth) {
    if (lo >= hi) return;

    // 작은 구간/깊은 재귀는 삽입 정렬로 마무리 (자식 런치 폭주 방지)
    if (hi - lo < 64 || depth >= 12) {
        for (int i = lo + 1; i <= hi; ++i) {
            int key = data[i], j = i - 1;
            while (j >= lo && data[j] > key) { data[j + 1] = data[j]; --j; }
            data[j + 1] = key;
        }
        return;
    }

    int pivot = data[hi], i = lo - 1;
    for (int j = lo; j < hi; ++j) if (data[j] < pivot) { ++i; dswap(data[i], data[j]); }
    dswap(data[i + 1], data[hi]);
    int p = i + 1;

    // 자식 커널 두 개를 각자 스트림에서 런치 (dynamic parallelism)
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    quicksortDP<<<1, 1, 0, s1>>>(data, lo, p - 1, depth + 1);
    quicksortDP<<<1, 1, 0, s2>>>(data, p + 1, hi, depth + 1);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 14);

    std::vector<int> h(n);
    unsigned s = 999u;
    for (int i = 0; i < n; ++i) { s = s * 1103515245u + 12345u; h[i] = (int)((s >> 16) & 0xffff); }
    std::vector<int> ref = h;
    std::sort(ref.begin(), ref.end());

    DeviceBuffer<int> d(n);
    d.copyFromHost(h.data());

    quicksortDP<<<1, 1>>>(d.data(), 0, n - 1, 0);   // 부모 런치 → 재귀 자식 런치
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> out(n);
    d.copyToHost(out.data());
    bool ok = (out == ref);
    std::printf("Dynamic Parallelism quicksort (n=%d): %s\n", n, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
