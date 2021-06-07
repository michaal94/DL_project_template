import os
import shutil
from datetime import datetime


'''
A couple of utilities I found useful
'''


# Make directories if path does not exists
# Input can be one path or list
def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)
    else:
        if not os.path.exists(paths):
            os.makedirs(paths)


# Simple file copying
def copy_file(src, dst):
    shutil.copyfile(src, dst)


# Copy directories (with content)
def copy_dirs(src_list, dst):
    for d in src_list:
        shutil.copytree(d, os.path.join(dst, os.path.basename(d)), symlinks=True)


# Invert dictionary
def invert_dict(d):
    return {v: k for k, v in d.items()}


# Add timestamp to the name
def timestamp_dir(logdir):
    main_dir, exp_dir = os.path.split(logdir)
    # Append 'timestamp' to the experiment directory name
    now = datetime.now()
    yy = now.year % 100
    m = now.month
    dd = now.day
    hh = now.hour
    mm = now.minute
    ss = now.second
    timestamp = "{:02d}{:02d}{:02d}_{:02d}{:02d}{:02d}".format(dd, m, yy, hh, mm, ss)
    exp_dir = "{}_{}".format(exp_dir, timestamp)
    logdir = os.path.join(main_dir, exp_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    return logdir
