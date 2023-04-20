cmake_minimum_required(VERSION 3.8)
project(distopt_interface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_definitions(-DNUM_INTRINSICS_PARAMS=1)
add_definitions(-DNUM_DISTORTION_PARAMS=2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(rclcpp REQUIRED HINTS "/opt/ros/humble/include/rclcpp/")
find_package(Ceres REQUIRED HINTS "/shared/lib/cmake/Ceres")
find_package(Eigen3 REQUIRED)
find_package(GTest CONFIG REQUIRED HINTS "/shared/lib/cmake/GTest")
find_package(Glog 0.4.0 EXACT REQUIRED HINTS "/shared/lib/cmake/glog") # For some reason Glog cannot be found (but dependency on ceres fixes it)
find_package(Gflags REQUIRED HINTS "/shared/lib/cmake/gflags")

include_directories(AFTER include)
include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${GFLAGS_INCLUDE_DIR})
set(CMAKE_INCLUDE_CURRENT_DIR ON)
# add_library(distopt
#       SHARED
# )

set(ADDED_LIBRARIES
  distributed_ba
  ${MPI_LIBRARIES}  
#  ${OPEN_MPI_LIBS}
  ${CERES_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  libstdc++fs.a
)

# look at how DBA did it
add_executable(main_decentral_async src/main_decentral_async.cpp)
target_link_libraries(main_decentral_async ${ADDED_LIBRARIES})

ament_target_dependencies(main_decentral_async rclcpp geometry_msgs)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


rosidl_generate_interfaces(${PROJECT_NAME}
	"msg/NodeData.msg"
  "msg/Residual.msg"
  "msg/FinishFlag.msg"
  "msg/Update.msg"
  "msg/Flag.msg"
  "msg/Counter.msg"
	DEPENDENCIES geometry_msgs
)

ament_package()


