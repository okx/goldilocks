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

$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -g -dc $< --output-file $@

.PHONY: clean

testscpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) tests/tests.cpp src/*.cpp -lgtest -lgmp -g -Wall -pthread -fopenmp -mavx2 -o $@

testsgpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ tests/tests.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o tests.o
	$(CXX) src/goldilocks_base_field.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) src/goldilocks_cubic_extension.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_cubic_extension.o
	$(CXX) utils/timer.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o timer.o
	$(NVCC) -D__USE_CUDA__ -Iutils -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/ntt_goldilocks.cu -arch=$(CUDA_ARCH) -g -dc --output-file ntt_goldilocks_gpu.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -g -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -g -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 tests/goldilocks_cubic_extension.cu -arch=$(CUDA_ARCH) -g -dc --output-file goldilocks_cubic_extension_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -g -o $@ tests.o timer.o goldilocks_base_field.o goldilocks_cubic_extension.o ntt_goldilocks_gpu.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o goldilocks_cubic_extension_gpu.o -lgtest -lgmp

tbtogpu: tests/test_merkle.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ tests/test_merkle.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o tests.o
	$(CXX) src/goldilocks_base_field.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(CXX) utils/timer.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o timer.o
	$(NVCC) -D__USE_CUDA__ -DFDEBUG -DGPU_TIMING -Iutils/ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -Xcompiler -O3 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -O3 -o $@ tests.o timer.o goldilocks_base_field.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o -lgtest -lgmp

tnttcpu: tests/test_ntt.cpp $(ALLSRCS)
	$(CXX) tests/test_ntt.cpp src/*.cpp -lgtest -lgmp -O3 -Wall -pthread -fopenmp -mavx2 -o $@

avxcpu: tests/test_poseidon.cpp $(ALLSRCS)
	$(CXX) tests/test_poseidon.cpp src/*.cpp -lgtest -lgmp -O3 -Wall -pthread -fopenmp -mavx2 -o $@
	./avxcpu

CXXFLAGS := -fPIC -O3 -Wall -pthread -fopenmp -mavx2
AVXFLAG := -Xcompiler -mavx2
# Verify if AVX-512 is supported
AVX512_SUPPORTED := $(shell cat /proc/cpuinfo | grep -E 'avx512' -m 1)

ifneq ($(AVX512_SUPPORTED),)
	CXXFLAGS += -mavx512f -D__AVX512__
	AVXFLAG += -Xcompiler -mavx512f -D__AVX512__
endif

fullgpu: tests/test_poseidon.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ tests/test_poseidon.cpp $(CXXFLAGS) -c
	$(CXX) src/goldilocks_base_field.cpp $(CXXFLAGS) -c -o goldilocks_base_field.o
	$(CXX) utils/timer.cpp $(CXXFLAGS) -c -o timer.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp $(CXXFLAGS) -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Iutils/ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 $(AVXFLAG) -Xcompiler -O3 src/ntt_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file ntt_goldilocks_gpu.o
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 $(AVXFLAG) src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -O3 -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -O3 -o $@ test_poseidon.o timer.o goldilocks_base_field.o ntt_goldilocks_gpu.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o -lgtest -lgmp

runfullgpu: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full

runfullcpu: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full_cpu

runcopy: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.copy

tnttgpu: tests/test_ntt.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ tests/test_ntt.cpp -O3 -pthread -fopenmp -mavx2 -c
	$(CXX) src/goldilocks_base_field.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) utils/timer.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o timer.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -Iutils/ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -mavx2 -Xcompiler -O3 src/ntt_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file ntt_goldilocks_gpu.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -O3 -Xcompiler -mavx2 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -O3 -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -O3 -o $@ test_ntt.o timer.o goldilocks_base_field.o ntt_goldilocks_gpu.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o -lgtest -lgmp

runtestscpu: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestsgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

benchcpu: benchs/bench.cpp $(ALLSRCS)
	$(CXX) benchs/bench.cpp src/*.cpp -lbenchmark -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -O3 -o $@

benchgpu: benchs/bench.cpp $(ALLSRCS)
	$(CXX) -D__USE_CUDA__ benchs/bench.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o tests.o
	$(CXX) src/goldilocks_base_field.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_base_field.o
	$(CXX) src/goldilocks_cubic_extension.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o goldilocks_cubic_extension.o
	$(CXX) -D__USE_CUDA__ src/poseidon_goldilocks.cpp -fPIC -O3 -Wall -pthread -fopenmp -mavx2 -c -o poseidon_goldilocks.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/ntt_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file ntt_goldilocks_gpu.o
	$(NVCC) -D__USE_CUDA__ -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 src/poseidon_goldilocks.cu -arch=$(CUDA_ARCH) -dc --output-file poseidon_goldilocks_gpu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -O3 -o $@ tests.o goldilocks_base_field.o goldilocks_cubic_extension.o ntt_goldilocks_gpu.o poseidon_goldilocks.o poseidon_goldilocks_gpu.o -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p
