import os
import glob
import sys
import random
import shutil

DATA_DIR = sys.argv[1]

if __name__ == "__main__":
    data_files = glob.glob(f"{DATA_DIR}/*.data")
    random.shuffle(data_files)
    n_dev = 2
    n_test = 5
    n_train = len(data_files) - n_dev - n_test
    sets = [
        ("train", data_files[: n_train]),
        ("dev", data_files[n_train : n_train + n_dev]),
        ("test", data_files[n_train + n_dev :])
    ]
    for dir_basename, dir_contents in sets:
        new_dir = DATA_DIR.replace(DATA_DIR.split("/")[-1], dir_basename)
        if os.path.isdir(new_dir):
            shutil.rmtree(new_dir)
        os.mkdir(new_dir)
        for f in dir_contents:
            os.symlink(f, f.replace(DATA_DIR, new_dir))  # populate the data subset directory with symlinks
