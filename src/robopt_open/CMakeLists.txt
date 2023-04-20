cmake_minimum_required(VERSION 3.10)
project(robopt_open)


# add_definitions(-std=c++11 -Wno-enum-compare)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake_gtest REQUIRED)
find_package(Ceres REQUIRED HINTS "/shared/lib/cmake/Ceres")
find_package(Eigen3 REQUIRED)
find_package(GTest CONFIG REQUIRED HINTS "/shared/lib/cmake/GTest")
find_package(Glog 0.4.0 EXACT REQUIRED HINTS "/shared/lib/cmake/glog") # For some reason Glog cannot be found (but dependency on ceres fixes it)
find_package(Gflags REQUIRED HINTS "/shared/lib/cmake/gflags")
find_package(eigen_checks REQUIRED)

include_directories("include/robopt_open"
"include/")
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_PATH})

add_library(${PROJECT_NAME}
  SHARED
  src/local-parameterization/quaternion-local-param.cpp
  src/local-parameterization/quaternion-yaw-local-param.cpp
  src/posegraph-error/six-dof-between.cpp
  src/posegraph-error/four-dof-between.cpp
  src/posegraph-error/relative-distance.cpp
  src/imu-error/preintegration-base.cpp
  src/imu-error/preintegration-factor.cpp
  src/reprojection-error/triangulation.cpp)

ament_add_gtest(test_pose_quaternion_local_parameterization
  src/local-parameterization/test/test_pose_quaternion_local_parameterization.cpp)
target_link_libraries(test_pose_quaternion_local_parameterization
  ${PROJECT_NAME})

ament_add_gtest(test_pose_quaternion_yaw_local_parameterization
  src/local-parameterization/test/test_pose_quaternion_yaw_local_parameterization.cpp)
target_link_libraries(test_pose_quaternion_yaw_local_parameterization
  ${PROJECT_NAME})

ament_add_gtest(test_global_euclidean_test
  src/reprojection-error/test/test_global_euclidean.cpp)
target_link_libraries(test_global_euclidean_test ${PROJECT_NAME})

ament_add_gtest(test_local_euclidean_test
  src/reprojection-error/test/test_local_euclidean.cpp)
target_link_libraries(test_local_euclidean_test ${PROJECT_NAME})

ament_add_gtest(test_relative_euclidean_test
  src/reprojection-error/test/test_relative_euclidean.cpp)
target_link_libraries(test_relative_euclidean_test ${PROJECT_NAME})

ament_add_gtest(test_triangulation_test
  src/reprojection-error/test/test_triangulation.cpp)
target_link_libraries(test_triangulation_test ${PROJECT_NAME})

ament_add_gtest(test_smart_projection_test
  src/reprojection-error/test/test_smart_projection.cpp)
target_link_libraries(test_smart_projection_test ${PROJECT_NAME})


ament_add_gtest(test_six_dof_between_test
  src/posegraph-error/test/test_six_dof_between.cpp)
target_link_libraries(test_six_dof_between_test ${PROJECT_NAME})

ament_add_gtest(test_four_dof_between_test
  src/posegraph-error/test/test_four_dof_between.cpp)
target_link_libraries(test_four_dof_between_test ${PROJECT_NAME})

ament_add_gtest(test_relative_distance_test
  src/posegraph-error/test/test_relative_distance.cpp)
target_link_libraries(test_relative_distance_test ${PROJECT_NAME})

ament_add_gtest(test_integration_base_test
  src/imu-error/test/test_integration_base.cpp)
target_link_libraries(test_integration_base_test ${PROJECT_NAME})

ament_add_gtest(test_imu_preintegration_test
  src/imu-error/test/test_imu_preintegration.cpp)
target_link_libraries(test_imu_preintegration_test ${PROJECT_NAME})


ament_package()