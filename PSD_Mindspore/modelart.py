import sys
import subprocess
import os

os.chdir('/home/work/user-job-dir/code/')

os.system('pip install sklearn  Cython  open3d-python')
os.system('bash compile_op.sh')
os.system('python train.py ' + ' '.join(sys.argv[1:]))

