
import sys
import os.path
import shutil
from absl import app
from absl import flags
from absl.flags import argparse_flags

FLAGS = flags.FLAGS

flags.DEFINE_string('result_folder', '', 'The location where the data should be written to')
flags.DEFINE_string('data_folder', '', 'The location where the input (initial) data is stored')
flags.DEFINE_string('admm_type', '', 'The admm type that was used (central_sync, central_async or decentral_async)')
flags.DEFINE_integer('num_subgraphs', 0, 'The number of subgraphs of the problem')
flags.DEFINE_integer('max_iter_eval', 0, 'The umber of evaluation iterations')

# Add the remaining flags used in order to avoid crashes
flags.DEFINE_integer('write_out_iter', -1, 'Dummy')
flags.DEFINE_float('alpha_map_points', -1, 'Dummy')
flags.DEFINE_float('alpha_intrinsics', -1, 'Dummy')
flags.DEFINE_float('alpha_distortion', -1, 'Dummy')
flags.DEFINE_float('alpha_rotation', -1, 'Dummy')
flags.DEFINE_float('alpha_translation', -1, 'Dummy')
flags.DEFINE_boolean('self_adaptation', 0, 'Dummy')
flags.DEFINE_float('alpha', -1.0, 'Dummy')
flags.DEFINE_float('asynch_nu', -1.0, 'Dummy')
flags.DEFINE_integer('num_ceres_threads', -1, 'Dummy')
flags.DEFINE_integer('num_ceres_iter', -1, 'Dummy')
flags.DEFINE_integer('num_admm_iter', -1, 'Dummy')
flags.DEFINE_integer('num_async', -1, 'Dummy')
flags.DEFINE_float('observation_sigma', -1.0, 'Dummy')

def main(argv):
  argv = FLAGS(argv)
  if (FLAGS.result_folder is None):
    sys.exit(1)
  # First check for the asynchronous methods whether the _0 iteration exists for
  # for all graphs, otherwise copy the initial state
  if (FLAGS.admm_type == "central_sync"):
    appendix = "cent_sync_"
  elif (FLAGS.admm_type == "central_async"):
    appendix = "cent_async_"
  elif (FLAGS.admm_type == "decentral_async"):
    appendix = "decent_async_"
  else :
    sys.exit(1)

  if (FLAGS.admm_type != "central_sync"):
    for graph_id in range(FLAGS.num_subgraphs):
      sub_folder = FLAGS.result_folder + "/Graph_" + str(graph_id)
      curr_frame_file = sub_folder + "/frames_opt_" + appendix + "0.csv"
      curr_map_point_file = sub_folder + "/map_points_opt_" + appendix + "0.csv"
      if not os.path.isfile(curr_frame_file): 
        print "replaces missing initial file"
        copy_frame_file = FLAGS.data_folder + "/Graph_" + str(graph_id) + "/frames.csv"
        shutil.copy(copy_frame_file, curr_frame_file)
        copy_map_point_file = FLAGS.data_folder + "/Graph_" + str(graph_id) + "/map_points.csv"
        shutil.copy(copy_map_point_file, curr_map_point_file)
    
  for graph_id in range(FLAGS.num_subgraphs):
    sub_folder = FLAGS.result_folder + "/Graph_" + str(graph_id)
    for file_id in range(1,FLAGS.max_iter_eval + 1):
      curr_frame_file = sub_folder + "/frames_opt_" + appendix + str(file_id) + ".csv"
      curr_map_point_file = sub_folder + "/map_points_opt_" + appendix + str(file_id) + ".csv"
      if not os.path.isfile(curr_frame_file):
        copy_frame_file = sub_folder + "/frames_opt_" + appendix + str(file_id - 1) + ".csv"
        shutil.copy(copy_frame_file, curr_frame_file)
        copy_map_point_file = sub_folder + "/map_points_opt_" + appendix + str(file_id - 1) + ".csv"
        shutil.copy(copy_map_point_file, curr_map_point_file)
  

if __name__ == '__main__':
  main(sys.argv)
