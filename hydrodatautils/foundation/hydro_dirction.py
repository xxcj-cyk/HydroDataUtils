import os
import platform

def get_origin_dir(dataset_name=None):
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        origin_dir = os.path.join("E:\\", "Takusan_no_Code", "Dataset", "Original_Dataset")
    else:
        origin_dir = os.path.join(home, "Dataset", "Original_Dataset")

    if dataset_name:
        origin_dir = os.path.join(origin_dir, dataset_name)

    if not os.path.exists(origin_dir):
        os.makedirs(origin_dir)

    return origin_dir

def get_cache_dir(dataset_name=None):
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        cache_dir = os.path.join("E:\\", "Takusan_no_Code", "Dataset", "Interim_Dataset")
    else:
        cache_dir = os.path.join(home, "Dataset", "Interim_Dataset")

    if dataset_name:
        cache_dir = os.path.join(cache_dir, dataset_name)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir

def get_export_dir(dataset_name=None):
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        export_dir = os.path.join("E:\\", "Takusan_no_Code", "Dataset", "Processed_Dataset")
    else:
        export_dir = os.path.join(home, "Dataset", "Processed_Dataset")

    if dataset_name:
        export_dir = os.path.join(export_dir, dataset_name)

    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    return export_dir