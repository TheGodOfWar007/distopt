#!/bin/bash

# Wait for apt to be ready 
while sudo fuser /var/lib/dpkg/lock >/dev/null 2>&1 ; do
  echo "Waiting for apt 1"
  sleep 1
done
while sudo fuser /var/lib/apt/lists/lock >/dev/null 2>&1 ; do
  echo "Waiting for apt 2"
  sleep 1
done
if [ -f /var/log/unattended-upgrades/unattended-upgrades.log ]; then
  while sudo fuser /var/log/unattended-upgrades/unattended-upgrades.log >/dev/null 2>&1 ; do
  echo "idk"
    sleep 1
  done
fi

echo "Starts the installation Process"


sudo apt install libeigen3-dev -y
sudo apt install cmake -y
sudo apt install libatlas-base-dev -y
sudo apt-get install libblas-dev liblapack-dev -y
#sudo apt install libsuitesparse-dev -y
sudo apt install unzip -y

export PATH="~/shared:$PATH"

cd /shared

# Install Suitesparse
# TODO: if this works we need to dismiss the compilation of the graphblas as this takes a shitload of time and is not used!

# First install openBLAS w/ apt install
if [ ! -d "/shared/SuiteSparse" ]; then
  wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v5.1.2.zip
  unzip v5.1.2.zip
  rm v5.1.2.zip 
  mv SuiteSparse-5.1.2 SuiteSparse # cd SuiteSparse
  make library
  make install INSTALL=/shared
fi

# Install gtest
if [ ! -d "/shared/googletest" ]; then
  wget https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
  unzip release-1.11.0.zip
  rm release-1.11.0.zip
  mv googletest-release-1.11.0 googletest
  cd /shared/googletest
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/shared
  make -j4
  make install
fi

cd /shared

# Install gflags
if [ ! -d "/shared/gflags" ]; then
  wget https://github.com/gflags/gflags/archive/refs/tags/v2.2.2.zip
  unzip v2.2.2.zip
  rm v2.2.2.zip
  mv gflags-2.2.2 gflags
  cd /shared/gflags
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/shared
  make -j4
  make install
fi

cd /shared

# Install glog
# sudo apt install glog
if [ ! -d "/shared/glog" ]; then
  wget https://github.com/google/glog/archive/refs/tags/v0.4.0.zip
  unzip v0.4.0.zip
  rm v0.4.0.zip
  mv glog-0.4.0 glog
  cd /shared/glog
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/shared
  make -j4
  make install
fi

cd /shared

# Install Ceres

# ceres/cmake/FindTBB.cmake line 434
# file(STRINGS
#      "${TBB_INCLUDE_DIR}/oneapi/tbb/version.h"
#      TBB_VERSION_CONTENTS
#      REGEX "VERSION")

if [ ! -d "/shared/ceres" ]; then
  wget ceres-solver.org/ceres-solver-2.0.0.tar.gz
  tar zxf ceres-solver-2.0.0.tar.gz
  rm ceres-solver-2.0.0.tar.gz
  mv ceres-solver-2.0.0 ceres
  cd ceres
  mkdir build && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/shared -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_TESTING=OFF -DSUITESPARSE_INCLUDE_DIR_HINTS=/shared/include -DSUITESPARSE_LIBRARY_DIR_HINTS=/shared/lib
  make -j4
  make install
fi
