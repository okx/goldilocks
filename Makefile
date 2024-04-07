#$(shell ./configure.sh)
#$(shell sleep 2)
include CudaArch.mk

TARGET_EXEC := example

BUILD_DIR := ./build
BUILD_DIR_GPU := ./build_gpu
SRC_DIRS := ./src
TEST_DIRS := ./test

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

OPTFLAG :=

# Debug build flags
ifeq ($(dbg),1)
      CXXFLAGS += -g
      OPTFLAG += -g
else
      CXXFLAGS += -O3
      OPTFLAG += -O3
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
	$(NVCC) -D__USE_CUDA__ -DGPU_TIMING $(OPTFLAG) -Iutils -Xcompiler -fopenmp -Xcompiler -fPIC -Xcompiler -mavx2 -arch=$(CUDA_ARCH) -dc $< --output-file $@

.PHONY: clean

tbtogpu: $(BUILD_DIR_GPU)/tests/test_merkle.cpp.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/utils/timer.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

tnttcpu: tests/test_ntt.cpp $(ALLSRCS)
	$(CXX) tests/test_ntt.cpp src/*.cpp -lgtest -lgmp -O3 -Wall -pthread -fopenmp -mavx2 -o $@

tnttgpu: $(BUILD_DIR_GPU)/tests/test_ntt.cpp.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o $(BUILD_DIR_GPU)/utils/timer.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

testscpu: tests/tests.cpp $(ALLSRCS)
	$(CXX) tests/tests.cpp src/*.cpp -lgtest -lgmp -g -Wall -pthread -fopenmp -mavx2 -o $@

testsgpu: $(BUILD_DIR_GPU)/tests/tests.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR)/utils/timer.cpp.o $(BUILD_DIR)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o $(BUILD_DIR_GPU)/utils/cuda_utils.cu.o $(BUILD_DIR_GPU)/tests/goldilocks_cubic_extension.cu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runtestscpu: testscpu
	./testscpu --gtest_filter=GOLDILOCKS_TEST.merkletree_seq

runtestsgpu: testsgpu
	./testsgpu --gtest_filter=GOLDILOCKS_TEST.merkletree_cuda

fullgpu: $(BUILD_DIR_GPU)/tests/test_poseidon.cu.o $(BUILD_DIR_GPU)/src/goldilocks_base_field.cpp.o  $(BUILD_DIR_GPU)/utils/timer.cpp.o  $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -DGPU_TIMING -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp

runfullgpu: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full0

runfullgpuv: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full1

runfullgpu2: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full2

runfullum: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full_um

runfullum2: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full_um2

runfullcpu: fullgpu
	./fullgpu --gtest_filter=GOLDILOCKS_TEST.full_cpu

benchcpu: $(BUILD_DIR)/
	$(CXX) benchs/bench.cpp src/*.cpp -lbenchmark -lpthread -lgmp  -std=c++17 -Wall -pthread -fopenmp -mavx2 -O3 -o $@

benchgpu: $(BUILD_DIR_GPU)/benchs/bench.cpp.o $(BUILD_DIR)/src/goldilocks_base_field.cpp.o $(BUILD_DIR)/src/goldilocks_cubic_extension.cpp.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cpp.o $(BUILD_DIR_GPU)/src/ntt_goldilocks.cu.o $(BUILD_DIR_GPU)/src/poseidon_goldilocks.cu.o
	$(NVCC) -Xcompiler -fopenmp -arch=$(CUDA_ARCH) -o $@ $^ -lgtest -lgmp -lbenchmark

runbenchcpu: benchcpu
	./benchcpu --benchmark_filter=MERKLETREE_BENCH_AVX

runbenchgpu: benchgpu
	./benchgpu --benchmark_filter=MERKLETREE_BENCH_CUDA

clean:
	$(RM) -r $(BUILD_DIR)
	$(RM) -r $(BUILD_DIR_GPU)

-include $(DEPS)

MKDIR_P ?= mkdir -p
