cmake_minimum_required(VERSION 3.10)
project(distributed_ba VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNUM_INTRINSICS_PARAMS=1)
add_definitions(-DNUM_DISTORTION_PARAMS=2)

find_package(ament_cmake REQUIRED)
find_package(Ceres REQUIRED HINTS "/shared/lib/cmake/Ceres")
find_package(Eigen3 REQUIRED)
find_package(GTest CONFIG REQUIRED HINTS "/shared/lib/cmake/GTest")
find_package(Glog 0.4.0 EXACT REQUIRED HINTS "/shared/lib/cmake/glog") # For some reason Glog cannot be found (but dependency on ceres fixes it)
find_package(Gflags REQUIRED HINTS "/shared/lib/cmake/gflags")
find_package(MPI REQUIRED)

include_directories(AFTER include)
include_directories(${EIGEN3_INCLUDE_DIRS})
#set(OPEN_MPI_INCLUDE_DIRS /home/mpiuser/Tools/open_mpi/include)
#set(OPEN_MPI_LIBS /home/mpiuser/Tools/open_mpi/lib/libmpi.so)
#include_directories(${OPEN_MPI_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

add_library(distributed_ba
     SHARED
    src/distributed_bundle_adjustment/frame.cpp
    src/distributed_bundle_adjustment/map_point.cpp
    src/distributed_bundle_adjustment/data.cpp
    src/distributed_bundle_adjustment/optimization.cpp
    src/distributed_bundle_adjustment/distortion.cpp
    src/distributed_bundle_adjustment/rad_distortion.cpp
    src/distributed_bundle_adjustment/radtan_distortion.cpp
    src/distributed_bundle_adjustment/equidistant_distortion.cpp
    src/distributed_bundle_adjustment/camera.cpp
    src/distributed_bundle_adjustment/simple_pinhole_camera.cpp
    src/distributed_bundle_adjustment/pinhole_camera.cpp
    src/distributed_bundle_adjustment/cost_functions/central_consensus.cpp
    src/distributed_bundle_adjustment/cost_functions/decentral_consensus.cpp
    src/distributed_bundle_adjustment/central_communication.cpp
    src/distributed_bundle_adjustment/central_async_communication.cpp
    src/distributed_bundle_adjustment/asynchronous_communication.cpp
)

set(ADDED_LIBRARIES
  distributed_ba
  ${MPI_LIBRARIES}  
#  ${OPEN_MPI_LIBS}
  ${CERES_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  libstdc++fs.a)

add_executable(sync_ba src/main_sync.cpp)
target_link_libraries(sync_ba ${ADDED_LIBRARIES})

add_executable(central_async_ba src/main_central_async.cpp)
target_link_libraries(central_async_ba ${ADDED_LIBRARIES})

add_executable(async_ba src/main_async.cpp)
target_link_libraries(async_ba ${ADDED_LIBRARIES})

add_executable(central_ba src/main_central_ba.cpp)
target_link_libraries(central_ba ${ADDED_LIBRARIES})

add_executable(evaluation src/main_eval.cpp)
target_link_libraries(evaluation ${ADDED_LIBRARIES})

add_executable(mpi_evaluation src/main_mpi_eval.cpp)
target_link_libraries(mpi_evaluation ${ADDED_LIBRARIES})

add_executable(test_distortion test/test_distortion.cpp)
target_link_libraries(test_distortion GTest::gtest GTest::gmock ${ADDED_LIBRARIES})

add_executable(test_camera test/test_camera.cpp)
target_link_libraries(test_camera GTest::gtest GTest::gmock ${ADDED_LIBRARIES})

add_executable(test_reprojection test/test_reprojection_error.cpp)
target_link_libraries(test_reprojection GTest::gtest GTest::gmock ${ADDED_LIBRARIES})

add_executable(test_consensus test/test_consensus_error.cpp)
target_link_libraries(test_consensus GTest::gtest GTest::gmock ${ADDED_LIBRARIES})

ament_package()