cmake_minimum_required(VERSION 3.0.0)
project(demo VERSION 0.1.0)
set(CMAKE_INSTALL_PREFIX install)
set(CMAKE_BUILD_TYPE Release)

#
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
     message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

#
# CUDA
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# OpenCV
find_package(OpenCV 3.4.8 REQUIRED PATHS /root/PLATFORM/3rdparty/opencv348)
message(${OpenCV_VERSION})
message(${OpenCV_LIBS})
include_directories( ${OpenCV_INCLUDE_DIRS})

# TensorRT
set(TENSORRT_ROOT /root/PLATFORM/3rdparty/TensorRT-7.0.0.11)
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

# fastrt
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/extra/fastrt/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/extra/fastrt)
SET(FASTRTENGINE_LIB ${CMAKE_CURRENT_SOURCE_DIR}/lib/extra/fastrt/libFastRTEngine.so.0)

#
add_subdirectory(lib)

#
include_directories(lib)

#
aux_source_directory(. SRC)
add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(demo reid nvinfer)

#install
install(FILES main.cpp DESTINATION .)


