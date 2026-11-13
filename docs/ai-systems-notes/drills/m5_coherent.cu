// Module 5 drill: how does this GPU reach CPU memory?
//
//   1. capability report: coherence, NUMA, VMM, fabric-handle attributes
//   2. copy-engine bandwidth: cudaMemcpy H2D/D2H from pageable vs pinned memory
//   3. SM bandwidth: a kernel that reads CPU memory directly (no copy engine)
//        a) cudaHostAlloc'd (pinned, mapped) memory  -> works everywhere (zero-copy)
//        b) plain malloc'd memory                      -> needs HMM or ATS (Grace, GB10)
//   4. fabric handle probe: can this process cuMemCreate a FABRIC-exportable allocation?
//      (on GB200/GB300 that only works when IMEX + a channel are set up)
//
// Build + run:  bash drills/build_m5.sh /tmp/m5_coherent && /tmp/m5_coherent
//               (or: bash drills/m5_rack.sh bw)
// Optional arg: buffer size in MiB (default 1024).
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("  %s -> %s\n", #x, cudaGetErrorString(e_)); return -1.0; } } while (0)

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Grid-stride read of 16-byte vectors; accumulates so the compiler can't drop loads.
__global__ void read_kernel(const uint4* __restrict__ p, size_t n, unsigned long long* out) {
  unsigned int acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) atomicAdd(out, 1ull);
}

static double copy_bw(void* host, void* dev, size_t bytes, cudaMemcpyKind kind) {
  void* dst = kind == cudaMemcpyHostToDevice ? dev : host;
  void* src = kind == cudaMemcpyHostToDevice ? host : dev;
  CK(cudaMemcpy(dst, src, bytes, kind));  // warm-up (and first-touch for pageable)
  CK(cudaDeviceSynchronize());
  double best = 1e30;
  for (int r = 0; r < 5; r++) {
    double t = now();
    CK(cudaMemcpy(dst, src, bytes, kind));
    CK(cudaDeviceSynchronize());
    t = now() - t; if (t < best) best = t;
  }
  return bytes / best / 1e9;
}

static double kernel_read_bw(const void* p, size_t bytes, int sms) {
  unsigned long long* out; CK(cudaMalloc(&out, sizeof(*out)));
  size_t n = bytes / sizeof(uint4);
  read_kernel<<<sms * 8, 512>>>((const uint4*)p, n, out);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("  kernel failed: %s\n", cudaGetErrorString(e)); return -1.0; }
  double best = 1e30;
  for (int r = 0; r < 5; r++) {
    double t = now();
    read_kernel<<<sms * 8, 512>>>((const uint4*)p, n, out);
    CK(cudaDeviceSynchronize());
    t = now() - t; if (t < best) best = t;
  }
  cudaFree(out);
  return bytes / best / 1e9;
}

static int attr(CUdevice d, CUdevice_attribute a) { int v = -99; cuDeviceGetAttribute(&v, a, d); return v; }

int main(int argc, char** argv) {
  size_t mib = argc > 1 ? strtoull(argv[1], 0, 10) : 1024;
  size_t bytes = mib << 20;
  if (cuInit(0) != CUDA_SUCCESS) { printf("cuInit failed: no usable GPU/driver\n"); return 1; }
  CUdevice d; cuDeviceGet(&d, 0);
  char name[256]; cuDeviceGetName(name, sizeof name, d);
  int drv = 0; cuDriverGetVersion(&drv);
  int sms = attr(d, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
  cudaSetDevice(0);

  printf("==== 1. capability report: %s (driver API %d.%d) ====\n", name, drv / 1000, (drv % 1000) / 10);
  struct { CUdevice_attribute a; const char* n; const char* m; } A[] = {
    {CU_DEVICE_ATTRIBUTE_INTEGRATED, "INTEGRATED", "GPU shares the CPU's DRAM (GB10 Spark = 1, GB200 = 0)"},
    {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS, "PAGEABLE_MEMORY_ACCESS", "kernels may dereference malloc/mmap pointers"},
    {CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES, "..._USES_HOST_PAGE_TABLES", "1 = ATS: GPU walks the CPU's page tables (Grace)"},
    {CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED, "HOST_NATIVE_ATOMIC_SUPPORTED", "GPU atomics on CPU memory without faults (C2C)"},
    {CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, "CONCURRENT_MANAGED_ACCESS", "CPU+GPU may touch managed memory at once"},
    {CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST", "CPU loads GPU-resident managed memory directly"},
    {CU_DEVICE_ATTRIBUTE_NUMA_CONFIG, "NUMA_CONFIG", "1 = GPU memory is a NUMA node (NUMA mode); 0 = not (PCIe, or CDMM)"},
    {CU_DEVICE_ATTRIBUTE_NUMA_ID, "NUMA_ID", "NUMA node id of the GPU's own memory (-1 if none)"},
    {CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, "HOST_NUMA_ID", "closest CPU NUMA node"},
    {CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, "VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED", "cuMemCreate/cuMemMap"},
    {CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, "HANDLE_TYPE_FABRIC_SUPPORTED", "driver can export FABRIC handles (MNNVL)"},
    {CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, "MULTICAST_SUPPORTED", "cuMulticastCreate (NVLS; needs NVSwitch)"},
    {CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, "HOST_NUMA_VMM_SUPPORTED", "cuMemCreate on a host NUMA node (EGM-style)"},
    {CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, "GPU_DIRECT_RDMA_SUPPORTED", "NIC may DMA to cudaMalloc memory"},
  };
  for (auto& x : A) printf("  %-38s %4d   %s\n", x.n, attr(d, x.a), x.m);
  int pageable = attr(d, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS);
  int ats = attr(d, CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES);
  printf("  => %s\n", ats ? "hardware-coherent (ATS): one page table for CPU and GPU"
                     : pageable ? "software coherence (HMM): GPU faults migrate pages"
                     : "no pageable access: GPU reaches CPU memory only via pinned/managed memory");

  printf("\n==== 2. copy engine: cudaMemcpy, %zu MiB, best of 5 ====\n", mib);
  void* dev; if (cudaMalloc(&dev, bytes) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
  char* pageable_buf = (char*)malloc(bytes); memset(pageable_buf, 1, bytes);
  void* pinned; cudaHostAlloc(&pinned, bytes, cudaHostAllocMapped); memset(pinned, 1, bytes);
  printf("  H2D pageable %7.1f GB/s    pinned %7.1f GB/s\n",
         copy_bw(pageable_buf, dev, bytes, cudaMemcpyHostToDevice), copy_bw(pinned, dev, bytes, cudaMemcpyHostToDevice));
  printf("  D2H pageable %7.1f GB/s    pinned %7.1f GB/s\n",
         copy_bw(pageable_buf, dev, bytes, cudaMemcpyDeviceToHost), copy_bw(pinned, dev, bytes, cudaMemcpyDeviceToHost));

  printf("\n==== 3. SMs reading CPU memory directly (no copy), %zu MiB ====\n", mib);
  printf("  GPU memory (reference)       %7.1f GB/s\n", kernel_read_bw(dev, bytes, sms));
  void* mapped; cudaHostGetDevicePointer(&mapped, pinned, 0);
  printf("  cudaHostAlloc (zero-copy)    %7.1f GB/s\n", kernel_read_bw(mapped, bytes, sms));
  if (pageable) printf("  plain malloc (HMM/ATS)       %7.1f GB/s\n", kernel_read_bw(pageable_buf, bytes, sms));
  else          printf("  plain malloc                 skipped: PAGEABLE_MEMORY_ACCESS = 0 (the kernel would fault)\n");

  printf("\n==== 4. fabric handle probe (multi-node NVLink) ====\n");
  if (attr(d, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED) != 1) {
    printf("  driver reports no FABRIC handle support\n");
  } else {
    CUmemAllocationProp prop; memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    size_t gran = 0; cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    CUmemGenericAllocationHandle h; CUresult r = cuMemCreate(&h, gran, &prop, 0);
    const char* s = 0; cuGetErrorString(r, &s);
    if (r == CUDA_SUCCESS) {
      CUmemFabricHandle fh; CUresult r2 = cuMemExportToShareableHandle(&fh, h, CU_MEM_HANDLE_TYPE_FABRIC, 0);
      cuGetErrorString(r2, &s);
      printf("  cuMemCreate(FABRIC) OK, export -> %s (%zu-byte handle you could send over TCP)\n", s, sizeof fh);
      cuMemRelease(h);
    } else {
      printf("  cuMemCreate(FABRIC) failed: %s\n", s);
      printf("  the attribute above only says the driver knows the handle type; creating one needs an IMEX\n");
      printf("  channel this process can open (ls -l /dev/nvidia-caps-imex-channels/). Expected to fail on a\n");
      printf("  single box; on GB200/GB300 a failure here = IMEX/channel setup problem (NCCL MNNVL will fail too)\n");
    }
  }
  cudaFree(dev); cudaFreeHost(pinned); free(pageable_buf);
  return 0;
}
