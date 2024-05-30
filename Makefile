TARGET_EXEC := example

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIRS := ./src
TEST_DIRS := ./test

include CudaArch.mk

LIBOMP := $(shell find /usr/lib/llvm-* -name "libomp.so" | sed 's/libomp.so//')
ifndef LIBOMP
$(error LIBOMP is not set, you need to install libomp-dev)
endif

#CXX := mpiCC
CXX = g++
CXXFLAGS := -std=c++17 -Wall -pthread -fopenmp
LDFLAGS := -lpthread -lgmp -lstdc++ -lgmpxx -lbenchmark
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

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.cu)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)
ALLSRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.asm -or -name *.hpp -or -name *.cu -or -name *.cuh)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -mavx2

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(OBJS) $(CXXFLAGS) -o $@ $(LDFLAGS)

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

# c++ source with CUDA support
$(BUILD_DIR_GPU)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) -D__USE_CUDA__ -mavx2 $(CFLAGS) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

$(BUILD_DIR_GPU)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING -Iutils -Xcompiler -O3 -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -dc $< --output-file $@

.PHONY: clean

testscpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) tests/tests.cpp src/*.cpp -lgtest -lgmp -O3 -Wall -pthread -fopenmp -mavx2 -o $@

testsgpu: $(BUILD_DIR_GPU)/tests/tests.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR)/utils/timer.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runtestscpu: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestsgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

full: $(BUILD_DIR_GPU)/tests/tests.cu.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o  $(BUILD_DIR_GPU)/utils/timer.cpp.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o  $(BUILD_DIR_GPU)/src/ntt_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runfullgpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_gpu

runfullcpu: full
	./full --gtest_filter=GOLDILOCKS_TEST.full_cpu

benchcpu: $(BUILD_DIR)/
	$(CXX) benchs/bench.cpp src/*.cpp -lbenchmark -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -O3 -o $@

benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -O3 -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

qtestavx: tests/test_vec_quick.cpp
	$(CXX) $(CXXFLAGS) -D__USE_AVX__ -D__NO_GMP__ -O3 -mavx2 $^ -o $@

qtestavx512: tests/test_vec_quick.cpp
	$(CXX) $(CXXFLAGS) -D__USE_AVX__ -D__AVX512__ -D__NO_GMP__ -O3 -mavx2 -mavx512f $^ -o $@

qtestsve: tests/test_vec_quick.cpp
	$(CXX) -D__USE_SVE__ -D__NO_GMP__ -O3 -march=armv9-a -mtune=neoverse-v1 $^ -o $@

benchavx: benchs/bench_vec_ops.cpp src/goldilocks_base_field_avx.hpp src/goldilocks_base_field_scalar.hpp
	$(CXX) $(CXXFLAGS) -D__USE_AVX__ -D__NO_GMP__ -O3 -mavx2 benchs/bench_vec_ops.cpp -fopenmp -o $@

benchavxpapi: benchs/bench_vec_ops.cpp src/goldilocks_base_field_sve.hpp src/goldilocks_base_field_scalar.hpp utils/papi.cpp utils/papi.hpp
	$(CXX) $(CXXFLAGS) utils/papi.cpp -c
	$(CXX) $(CXXFLAGS) -Iutils -D__USE_PAPI__ -D__USE_AVX__ -D__NO_GMP__ -O3 -mavx2 benchs/bench_vec_ops.cpp -fopenmp -c
	$(CXX) $(CXXFLAGS) -O3 papi.o bench_vec_ops.o -o $@ -lpapi

benchmtavx: benchs/bench_vec_merkle.cpp src/poseidon_goldilocks.cpp src/goldilocks_base_field.cpp
	$(CXX) $(CXXFLAGS) -Isrc -D__USE_AVX__ -O3 -mavx2 $^ -fopenmp -o $@ -lgmp

benchmtavx512: benchs/bench_vec_merkle.cpp src/poseidon_goldilocks.cpp src/goldilocks_base_field.cpp
	$(CXX) $(CXXFLAGS) -Isrc -D__USE_AVX__ -D__AVX512__ -O3 -mavx2 -mavx512f $^ -fopenmp -o $@ -lgmp

benchsve: benchs/bench_vec_ops.cpp src/goldilocks_base_field_sve.hpp src/goldilocks_base_field_scalar.hpp
	$(CXX) -D__USE_SVE__ -D__NO_GMP__ -O3 -march=armv9-a -mtune=neoverse-v1 benchs/bench_vec_ops.cpp -fopenmp -o $@

benchsvepapi: benchs/bench_vec_ops.cpp src/goldilocks_base_field_sve.hpp src/goldilocks_base_field_scalar.hpp utils/papi.cpp utils/papi.hpp
	$(CXX) utils/papi.cpp -c
	$(CXX) -Iutils -D__USE_PAPI__ -D__USE_SVE__ -D__NO_GMP__ -O3 -march=armv9-a -mtune=neoverse-v1 benchs/bench_vec_ops.cpp -fopenmp -c
	$(CXX) -O3 papi.o bench_vec_ops.o -o $@ -lpapi

benchmtsve: benchs/bench_vec_merkle.cpp src/poseidon_goldilocks_sve.cpp
	$(CXX) -D__USE_SVE__ -Isrc -O3 -march=armv9-a -mtune=neoverse-v1 $^ -fopenmp -o $@ -lgmp

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)
	$(RM) -f benchavx benchavxpapi benchmtavx qtestavx benchsve benchsvepapi benchmtsve qtestsve

-include $(DEPS)

MKDIR_P ?= mkdir -p
