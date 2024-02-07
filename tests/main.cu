#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdint.h>
#include "../src/cuda_utils.cuh"

__global__ void addOneToEachElement(uint64_t *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  printf("123\n");
  if (idx < N) {
    data[idx] = __brev(data[idx]);
  }
}

int main() {
  const u_int64_t N = (1<<3);
  uint64_t *h_data = (uint64_t*)malloc(N * sizeof(uint64_t)); // 分配ht memory
  uint64_t *d_data; // 定义device memory指针

  CHECKCUDAERR(cudaMalloc((void**)&d_data, N * sizeof(uint64_t))); // 在device memor上分配内存

  // 初始化host memory数据
  for (uint64_t i = 0; i < N; ++i) {
    h_data[i] = i;
  }
#ifdef __PRINT_LOG__
  for (int i = 0; i < N; ++i) {
    printf("%b ", h_data[i]);
  }
  printf("\n");
#endif

  struct timeval start, end;
  gettimeofday(&start, NULL);
  // 将数据从host memory拷贝到device memory
  CHECKCUDAERR(cudaMemcpy(d_data, h_data, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
  gettimeofday(&end, NULL);
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  long elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy to device elapsed: %ld ms\n", elapsed);
  // 在device上执行每个数字加一的操作
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  addOneToEachElement<<<numBlocks, blockSize>>>(d_data, N);
  // 将数据从device memory拷贝回host memory
  gettimeofday(&start, NULL);
  CHECKCUDAERR(cudaMemcpy(h_data, d_data, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy to host elapsed: %ld ms\n", elapsed);
  // 打印结果
#ifdef __PRINT_LOG__
  for (int i = 0; i < N; ++i) {
    printf("%b ", h_data[i]);
  }
  printf("\n");
#endif

  // 释放内存

  gettimeofday(&start, NULL);
  CHECKCUDAERR(cudaFree(d_data));
  free(h_data);
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy to device and free elapsed: %ld ms\n", elapsed);

  return 0;
}