#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdint.h>
#include "../src/cuda_utils.cuh"

__global__ void addOneToEachElement(uint64_t *data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    data[idx] = __brev(data[idx]);
  }
}

const u_int64_t N = (1<<23) * 88;
const uint64_t MAX_LOG_ITEMS = 8;

int test1(uint64_t* h_data) {
  uint64_t *d_data; // 定义device memory指针

  CHECKCUDAERR(cudaMalloc((void**)&d_data, N * sizeof(uint64_t))); // 在device memor上分配内存


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
  gettimeofday(&start, NULL);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  addOneToEachElement<<<numBlocks, blockSize>>>(d_data, N);
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("kernel elapsed: %ld ms\n", elapsed);
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
  for (int i = 0; i < N && i < MAX_LOG_ITEMS; ++i) {
    printf("%b ", h_data[N-1-i]);
  }
  printf("\n");
#endif

  // 释放内存

  gettimeofday(&start, NULL);
  CHECKCUDAERR(cudaFree(d_data));
  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy to device and free elapsed: %ld ms\n", elapsed);

  return 0;
}

int test2(uint64_t* um_data) {

  struct timeval start, end;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  CHECKCUDAERR(cudaMemPrefetchAsync((void*)um_data, N * sizeof(uint64_t), 0, stream));
  // 在device上执行每个数字加一的操作
  gettimeofday(&start, NULL);
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  addOneToEachElement<<<numBlocks, blockSize, 0, stream>>>(um_data, N);
  cudaStreamSynchronize(stream);
  gettimeofday(&end, NULL);
  long seconds = end.tv_sec - start.tv_sec;
  long microseconds = end.tv_usec - start.tv_usec;
  long elapsed = seconds*1000 + microseconds/1000;
  printf("kernel elapsed: %ld ms\n", elapsed);

  // 将数据从device memory拷贝回host memory
  gettimeofday(&start, NULL);
  CHECKCUDAERR(cudaMemPrefetchAsync((void*)um_data, N * sizeof(uint64_t), cudaCpuDeviceId, stream));
  cudaStreamSynchronize(stream);

  gettimeofday(&end, NULL);
  seconds = end.tv_sec - start.tv_sec;
  microseconds = end.tv_usec - start.tv_usec;
  elapsed = seconds*1000 + microseconds/1000;
  printf("memcpy to host elapsed: %ld ms\n", elapsed);
  // 打印结果
#ifdef __PRINT_LOG__
  for (int i = 0; i < N && i < MAX_LOG_ITEMS; ++i) {
    printf("%b ", h_data[N-1-i]);
  }
  printf("\n");
#endif

  return 0;
}

int main() {
  struct timeval start, end;

  {
    uint64_t *h_data = (uint64_t*)malloc(N * sizeof(uint64_t)); // 分配ht memory
    // 初始化host memory数据
    for (uint64_t i = 0; i < N; ++i) {
      h_data[i] = i;
    }
    gettimeofday(&start, NULL);
    test1(h_data);
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    long elapsed = seconds*1000 + microseconds/1000;
    printf("test1 elapsed: %ld ms\n", elapsed);
    free(h_data);
  }

  {
    uint64_t *um_data;
    cudaMallocManaged(&um_data, N * sizeof(uint64_t));
    // 初始化host memory数据
    for (uint64_t i = 0; i < N; ++i) {
      um_data[i] = i;
    }

    gettimeofday(&start, NULL);
    test2(um_data);
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    long elapsed = seconds*1000 + microseconds/1000;
    printf("test1 elapsed: %ld ms\n", elapsed);
    cudaFree(um_data);
  }
}