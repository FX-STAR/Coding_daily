cmake_minimum_required(VERSION 2.6)
project(Demo)

SET(CMAKE_C_COMPILER g++)
SET(CMAK_CXX_COMPILER g++)
add_compile_options(-std=c++11)
add_compile_options(-m64)

#set(CMAKE_CXX_STANDARD 11)

include_directories ( 
    /home/lfx/Tool/caffe/include
    /home/lfx/Tool/caffe/build/src
    /usr/local/include
    /home/lfx/Software/cuda-9.0/include
    )
add_executable(Demo demo.cpp)

target_link_libraries(Demo
    /home/lfx/Tool/caffe/build/lib/libcaffe.so 
    boost_system 
    glog
    )

find_package(OpenCV REQUIRED)
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    )

target_link_libraries(Demo 
    ${OpenCV_LIBS}
)

