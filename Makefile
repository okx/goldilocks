$(shell ./configure.sh)
include CudaArch.mk

TARGET_EXEC := example

BUILD_DIR := ./build
SRC_DIRS := ./src
TEST_DIRS := ./test

LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

#CXX := mpiCC
CXX = g++
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp
LDFLAGS := -lpthread -lgmp -lstdc++ -lomp -lgmpxx -lbenchmark -L$(LIBOMP)
ASFLAGS := -felf64

CC := gcc
NVCC := /usr/local/cuda/bin/nvcc

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
else
      CXXFLAGS += -O3
endif

### Establish the operating system name
KERNEL = $(shell uname -s)
ifneq ($(KERNEL),Linux)
 $(error "$(KERNEL), is not a valid kernel")
endif
ARCH = $(shell uname -m)
ifneq ($(ARCH),x86_64)
 $(error "$(ARCH), is not a valid architecture")
endif

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ALLSRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.hpp -or -name *.cu -or -name *.cuh)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cc.o: %.cc
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

.PHONY: clean

testscpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) tests/tests.cpp src/*.cpp -lgtest -lgmp -g -Wall -pthread -fopenmp -mavx2 -o $@

testsgpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ tests/tests.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o tests.o
	$(CXX) -D__USE_CUDA__ cryptography_cuda/cuda/util/all_gpus.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -Icryptography_cuda/depends/blst/src -I/usr/local/cuda/include -Icryptography_cuda/cuda/util -o util.o
	$(CXX) src/goldilocks_base_field.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) src/goldilocks_cubic_extension.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_cubic_extension.o
	$(CXX) -D__USE_CUDA__ src/ntt_goldilocks.cpp -fPIC  -g -Wall -pthread -fopenmp -mavx2 -c -Icryptography_cuda/depends/blst/src -I/usr/local/cuda/include -Icryptography_cuda/cuda -o ntt_goldilocks.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -DFEATURE_GOLDILOCKS -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 cryptography_cuda/src/lib.cu -Icryptography_cuda/cuda -Icryptography_cuda/depends/blst/src -arch=$(CUDA_ARCH) -g -dc --output-file ntt_gpu.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -g -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -g -o $@ tests.o util.o goldilocks_base_field.o goldilocks_cubic_extension.o ntt_goldilocks.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o ntt_gpu.o -lgtest -lgmp

runtestscpu: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestsgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

runnttcpu: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.extendePol_cpu

runnttgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.extendePol_cuda

runrp1: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.rp1

runrp2: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.rp2

runintt: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.intt

runinttgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.intt_cuda

benchcpu: benchs/bench.cpp $(ALLSRCS)
	$(CXX) benchs/bench.cpp src/*.cpp -lbenchmark -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -O3 -o $@

benchgpu: benchs/bench.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ benchs/bench.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o tests.o
	$(CXX) src/goldilocks_base_field.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) src/goldilocks_cubic_extension.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_cubic_extension.o
	$(CXX) src/ntt_goldilocks.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o ntt_goldilocks.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -O3 -o $@ tests.o goldilocks_base_field.o goldilocks_cubic_extension.o ntt_goldilocks.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
