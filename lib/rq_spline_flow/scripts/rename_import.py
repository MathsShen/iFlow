import os
import os.path as osp

import glob
dir_path = "../nn/*.py"
tar_path = "../_nn/"
if not osp.exists(tar_path):
    os.makedirs(tar_path)
py_list = glob.glob(dir_path)

import pdb

for py_file in py_list:
    py_name = py_file.split("/")[-1]
    w_fobj = open(osp.join(tar_path, py_name), 'w')
    with open(py_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("import utils"):
                new_line = "from lib.rq_spline_flow import utils\n"

            elif line.startswith("from nde"):
                new_line = line.replace("from nde", "from lib.rq_spline_flow")
            
            else:
                new_line = line

            w_fobj.write(new_line)
    
    w_fobj.close()

print("DONE.")
