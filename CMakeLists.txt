cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
set(CMAKE_SUPPRESS_REGENERATION true)
project(PointCloudCalibration)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/../bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/../bin)

if (!MSVC)
	add_compile_options(-O3 -Wall -Wno-strict-aliasing -Werror -Wextra -Wpedantic)
else()
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

option(BUILD_SHARED_LIBS "Build LASlib as DLL" OFF)

if (BUILD_SHARED_LIBS AND UNIX AND NOT APPLE)
	set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib/LASlib")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(LASlib/src)
include_directories(LASlib/inc)
include_directories(LASzip/src)
include_directories(include)

add_subdirectory(LASlib/src)

add_executable(calibrate src/main.cpp)

target_link_libraries(calibrate 
	LASlib 
	Eigen3::Eigen
)