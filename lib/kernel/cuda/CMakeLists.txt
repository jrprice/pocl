#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

include("bitcode_rules")

set(KERNEL_SOURCES ${SOURCES_WITHOUT_VML})

foreach(FILE atomics.cl get_global_offset.c get_work_dim.c)
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
endforeach()

foreach(FILE
  barrier.ll
  get_global_id.c get_global_size.c get_group_id.c
  get_local_id.c get_local_size.c get_num_groups.c
  printf.c
  )
  list(REMOVE_ITEM KERNEL_SOURCES "${FILE}")
  list(APPEND KERNEL_SOURCES "cuda/${FILE}")
endforeach()

list(APPEND KERNEL_SOURCES "cuda/nvvm_functions.ll")

# Select either NVPTX or NVPTX64
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set(LLVM_TARGET nvptx64)
else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  set(LLVM_TARGET nvptx)
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )


set(CLANG_FLAGS "-emit-llvm" "-target" "${LLVM_TARGET}" "-D_CL_DISABLE_HALF")

if(POCL_USE_FAKE_ADDR_SPACE_IDS)
  list(APPEND CLANG_FLAGS "-Xclang" "-ffake-address-space-map" "-DPOCL_USE_FAKE_ADDR_SPACE_IDS")
endif()

# Enable all extensions
set(KERNEL_CL_FLAGS ${KERNEL_CL_FLAGS} "-Xclang" "-cl-ext=all")

set(LLC_FLAGS "")
set(DEVICE_CL_FLAGS "-D__OPENCL_VERSION__=${CUDA_DEVICE_CL_VERSION}")
separate_arguments(CUDA_DEVICE_EXTENSIONS)
foreach(EXT ${CUDA_DEVICE_EXTENSIONS})
  set(DEVICE_CL_FLAGS "${DEVICE_CL_FLAGS} -D${EXT}")
endforeach()
separate_arguments(DEVICE_CL_FLAGS)

make_kernel_bc(KERNEL_BC "${LLVM_TARGET}" "BCs" ${KERNEL_SOURCES})

# just debug
message(STATUS "${LLVM_TARGET} Kernel BC: ${KERNEL_BC}")

list(APPEND KERNEL_BC_LIST "${KERNEL_BC}")
set(KERNEL_BC_LIST "${KERNEL_BC_LIST}" PARENT_SCOPE)

# a target is needed...
add_custom_target("kernel_${LLVM_TARGET}" DEPENDS ${KERNEL_BC})

list(APPEND KERNEL_TARGET_LIST "kernel_${LLVM_TARGET}")
set(KERNEL_TARGET_LIST "${KERNEL_TARGET_LIST}" PARENT_SCOPE)

install(FILES "${KERNEL_BC}"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR}")
