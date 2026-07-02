// gds_demo.cu
// 슬라이드: part5/chapter38 (Scan / Prefix Sum)
// GPUDirect Storage(cuFile) 데모: NVMe 파일을 GPU VRAM으로 직접 DMA 로드 후 스캔.
// 빌드: cmake -DUSE_GDS=ON ..   실행: ./gds_demo <int32_binary_file>
// 주의: libcufile + GDS 지원 스토리지 필요. Jetson은 compat 모드일 수 있어 성능/가용성이 다르다.
#include "../common/raii.cuh"
#include "scan_kernels.cuh"
#include "scan_metric.cuh"

#include <cufile.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <int32 binary file>\n", argv[0]);
        return 1;
    }
    const char* path = argv[1];

    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { std::perror("open"); return 1; }

    off_t bytes = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    int N = static_cast<int>(bytes / sizeof(int));
    int numSeg = N / SEG;
    if (numSeg < 1) { std::fprintf(stderr, "file too small (need >= %d ints)\n", SEG); close(fd); return 1; }
    N = numSeg * SEG;

    if (cuFileDriverOpen().err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "cuFileDriverOpen failed\n"); close(fd); return 1;
    }

    CUfileDescr_t desc{};
    desc.handle.fd = fd;
    desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileHandle_t handle;
    if (cuFileHandleRegister(&handle, &desc).err != CU_FILE_SUCCESS) {
        std::fprintf(stderr, "cuFileHandleRegister failed\n"); return 1;
    }

    DeviceBuffer<int> d_in(N), d_out(N);
    cuFileBufRegister(d_in.data(), N * sizeof(int), 0);

    // NVMe → GPU VRAM 직접 읽기 (CPU 바운스 버퍼 없이)
    ssize_t rd = cuFileRead(handle, d_in.data(), N * sizeof(int), 0, 0);
    if (rd < 0) { std::fprintf(stderr, "cuFileRead failed\n"); return 1; }
    std::printf("GDS: %zd bytes 를 GPU로 직접 로드\n", rd);

    scanComposed<true><<<numSeg, SEG>>>(d_in.data(), d_out.data());
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // 검증: 호스트에서 파일을 다시 읽어 CPU 기준과 비교
    std::vector<int> h_in(N);
    {
        int f2 = open(path, O_RDONLY);
        if (f2 >= 0) { ssize_t n = read(f2, h_in.data(), N * sizeof(int)); (void)n; close(f2); }
    }
    auto ref = scanCpuReference(h_in, numSeg);
    std::vector<int> h_out(N);
    d_out.copyToHost(h_out.data());
    std::printf("검증: %s\n", (h_out == ref) ? "OK" : "FAIL");

    cuFileBufDeregister(d_in.data());
    cuFileHandleDeregister(handle);
    cuFileDriverClose();
    close(fd);
    return 0;
}
