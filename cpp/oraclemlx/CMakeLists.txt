cmake_minimum_required(VERSION 3.24)
project(oraclemlx LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MLX pinned at v0.30.6 in BOTH modes below.
#
# Building MLX from source requires the Metal offline compiler (full Xcode /
# `xcrun metal`). Machines with only Command Line Tools can't compile .metal
# kernels, so we support two modes:
#   1. Source (default when `xcrun --find metal` succeeds): FetchContent at the
#      pinned tag.
#   2. Prebuilt wheel: -DOMLX_MLX_ROOT=<site-packages>/mlx — the pip wheel
#      (`pip install mlx==0.30.6`) ships libmlx.dylib, mlx.metallib, headers,
#      and a CMake config. libmlx.dylib + mlx.metallib are copied next to
#      liboraclemlx.dylib so the bundle is self-contained.
option(OMLX_MLX_ROOT "Path to a prebuilt MLX (pip wheel mlx/ dir); empty = build from source" "")

if(OMLX_MLX_ROOT)
  find_package(MLX CONFIG REQUIRED PATHS ${OMLX_MLX_ROOT} NO_DEFAULT_PATH)
else()
  execute_process(COMMAND xcrun --find metal RESULT_VARIABLE METAL_MISSING
                  OUTPUT_QUIET ERROR_QUIET)
  if(METAL_MISSING)
    message(FATAL_ERROR
      "No Metal offline compiler (full Xcode required to build MLX from source). "
      "Either install Xcode, or pass the prebuilt wheel: pip install mlx==0.30.6 "
      "and configure with -DOMLX_MLX_ROOT=$(python -c 'import mlx,os;print(os.path.dirname(mlx.__file__))')")
  endif()
  include(FetchContent)
  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG v0.30.6
  )
  FetchContent_MakeAvailable(mlx)
endif()

add_library(oraclemlx SHARED oraclemlx.cpp)
target_link_libraries(oraclemlx PRIVATE mlx)
target_include_directories(oraclemlx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(oraclemlx PROPERTIES
  PUBLIC_HEADER oraclemlx.h
  BUILD_RPATH "@loader_path"
  INSTALL_RPATH "@loader_path")

if(OMLX_MLX_ROOT)
  # Self-contained output dir: mlx dylib + metallib next to liboraclemlx.dylib.
  add_custom_command(TARGET oraclemlx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OMLX_MLX_ROOT}/lib/libmlx.dylib
            ${OMLX_MLX_ROOT}/lib/mlx.metallib
            $<TARGET_FILE_DIR:oraclemlx>)
  install(FILES ${OMLX_MLX_ROOT}/lib/libmlx.dylib ${OMLX_MLX_ROOT}/lib/mlx.metallib
          DESTINATION lib)
endif()

install(TARGETS oraclemlx
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include)
