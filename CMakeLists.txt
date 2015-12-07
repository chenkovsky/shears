cmake_minimum_required(VERSION 2.8.4)
project(lmprune)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O0 -g -Wall -std=c++11")
INCLUDE_DIRECTORIES(../srilm-1.7.1/include/ docopt.cpp)
LINK_DIRECTORIES(../srilm-1.7.1/lib/macosx docopt.cpp)
add_executable(shears shears.cpp)
target_link_libraries(shears z oolm flm dstruct misc lattice iconv docopt_s boost_iostreams) # oolm flm dstruct are libraries from srilm
