import fnmatch
import os
import numpy as np


def get_lastest_file_in_a_dir(dir_path):
    """Get the last file in a directory

    Parameters
    ----------
    dir_path : str
        the directory

    Returns
    -------
    str
        the path of the weight file
    """
    pth_files_lst = [
        os.path.join(dir_path, file)
        for file in os.listdir(dir_path)
        if fnmatch.fnmatch(file, "*.pth")
    ]
    return get_latest_file_in_a_lst(pth_files_lst)


def get_latest_file_in_a_lst(lst):
    """get the latest file in a list

    Parameters
    ----------
    lst : list
        list of files

    Returns
    -------
    str
        the latest file
    """
    lst_ctime = [os.path.getctime(file) for file in lst]
    sort_idx = np.argsort(lst_ctime)
    return lst[sort_idx[-1]]