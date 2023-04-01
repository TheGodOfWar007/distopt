#!/bin/bash
# Parameters
#SBATCH --ntasks=17
#SBATCH --cpus-per-task=2

# Compile the actual code
if [ -d "/shared/distributed_bundle_adjustment/build" ]; then
  rm -r /shared/distributed_bundle_adjustment/build
fi
cd /shared/distributed_bundle_adjustment
mkdir build && cd build
cmake ..
make -j2

# Run the program
#srun ./sync_ba --flagfile=/shared/graph_data/Maplab/LEE_MH/Sixteen_Cut/config_flags_maplab
GLOG_logtostderr=1 mpirun ./sync_ba --flagfile=/shared/Data/Simulated/Sixteen_Cut/config_flags_sync

GLOG_logtostderr=1 mpirun ./mpi_evaluation --flagfile=/shared/Data/Simulated/Sixteen_Cut/config_flags_sync
