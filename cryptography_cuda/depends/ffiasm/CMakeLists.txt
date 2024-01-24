cmake_minimum_required(VERSION 3.22.1)

include(depends/cryptography_cuda/cmake/platform.cmake)

set(USE_ASM    ON CACHE BOOL "Use asm implementation for Fr and Fq")
set(USE_OPENMP ON CACHE BOOL "Use OpenMP")
OPTION(USE_CUDA "whether integrate with gpu" OFF) # Enabled by default


if(USE_CUDA)
    message("use cuda")
else()
    message("do not use cuda")
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 86)
endif()

project(ffiasm LANGUAGES CXX CUDA ASM)
set(CMAKE_CXX_STANDARD 17)


if (NOT EXISTS ${GMP_INCLUDE_FILE_FULLPATH})
    message("WARNING: ${GMP_INCLUDE_FILE_FULLPATH} is not found and so system ${GMP_INCLUDE_FILE} is used.")
endif()

if (NOT EXISTS ${GMP_LIB_FILE_FULLPATH})
    message("WARNING: ${GMP_LIB_FILE_FULLPATH} is not found and so system ${GMP_LIB_FILE} is used.")
    set(GMP_LIB gmp)
endif()

include_directories(BEFORE ${GMP_INCLUDE_DIR})


if(USE_OPENMP)
    find_package(OpenMP)

    if(OpenMP_CXX_FOUND)
        if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
            message("OpenMP is used")

        else()
            set(OpenMP_CXX_FOUND FALSE)
            message("OpenMP is not used")

        endif()
    endif()
endif()

find_package(CUDA REQUIRED)
enable_language("CUDA")

set(PROJECT_CUDA_SOURCE_DIR ${CMAKE_SOURCE_DIR}/depends/cryptography_cuda/cuda)

message("USE_ASM=: ${USE_ASM}")
message("CUDA_INCLUDE_DIRS=: ${CUDA_INCLUDE_DIRS}")
message("CUDA_TOOLKIT_ROOT_DIR=: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CMAKE_SOURCE_DIR=: ${CMAKE_SOURCE_DIR}")
message("CUDA_LIBRARY=: ${CUDA_LIBRARY}")
message("PROJECT_CUDA_SOURCE_DIR=: ${PROJECT_CUDA_SOURCE_DIR}")

include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_CUDA_SOURCE_DIR})
include_directories("depends/cryptography_cuda/depends/blst/src")  # the cuda library is dependent on blst

set(LIB_SOURCES
    depends/cryptography_cuda/src/lib.h
    depends/cryptography_cuda/src/lib.cu
    depends/cryptography_cuda/cuda/util/all_gpus.cpp
)
add_library(cryptography_cuda SHARED ${LIB_SOURCES} )
## todo: pass feature as environment variable, or as option
target_compile_definitions(cryptography_cuda PUBLIC -DFEATURE_BN128)


include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

link_libraries(${GMP_LIB})
add_definitions(${GMP_DEFINIONS})
add_definitions(-DUSE_ASM -DARCH_X86_64)
if(USE_CUDA)
add_definitions(-DUSE_CUDA)
endif()


set(NASM_FLAGS -felf64 -DPIC)
message("use custom command, NASM_FLAGS=${NASM_FLAGS}")

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/fq_asm.o
    COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/test/field/fq.asm -o fq_asm.o
    DEPENDS ${CMAKE_SOURCE_DIR}/test/field/fq.asm
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

add_custom_command(OUTPUT ${CMAKE_SOURCE_DIR}/build/fr_asm.o
    COMMAND nasm ${NASM_FLAGS} ${CMAKE_SOURCE_DIR}/test/field/fr.asm -o fr_asm.o
    DEPENDS ${CMAKE_SOURCE_DIR}/test/field/fr.asm
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/build)


set(FR_SOURCES
    test/field/fr.hpp
    test/field/fr.cpp
    )
set(FR_SOURCES ${FR_SOURCES} build/fr_asm.o)

add_library(fr STATIC ${FR_SOURCES})
set_target_properties(fr PROPERTIES POSITION_INDEPENDENT_CODE ON)

link_libraries(fr)



set(FQ_SOURCES
    test/field/fq.hpp
    test/field/fq.cpp
    )
set(FQ_SOURCES ${FQ_SOURCES} build/fq_asm.o)

add_library(fq STATIC ${FQ_SOURCES})
set_target_properties(fq PROPERTIES POSITION_INDEPENDENT_CODE ON)

link_libraries(fq)


add_definitions(-DUSE_OPENMP)

add_compile_options(${OpenMP_CXX_FLAGS})


set(TEST_FFT_SOURCES
    test/test_fft.cpp
    c/misc.cpp
    c/naf.cpp
    c/splitparstr.cpp
    c/alt_bn128.cpp
)

add_executable(test_fft ${TEST_FFT_SOURCES})
target_include_directories(
    test_fft
    PRIVATE
    depends/
    test/field
    c
)

target_link_libraries(test_fft cryptography_cuda)
target_link_libraries(test_fft OpenMP::OpenMP_CXX)
# install(TARGETS test_bn128 cryptography_cuda
#     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#     BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}/app
#     LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# install(FILES src/lib.h
#     DESTINATION ${CMAKE_INSTALL_PREFIX}/include)