import os
import platform


def get_cache_dirction(app_name="hydro", dataset_name=None):
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        cache_dir = os.path.join(home, "AppData", app_name, "temp")
    else:
        cache_dir = os.path.join(home, ".cache", app_name, "temp")

    if dataset_name:
        cache_dir = os.path.join(cache_dir, dataset_name)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir


def get_dataset_dirction(app_name="hydro", dataset_name=None):
    home = os.path.expanduser("~")
    system = platform.system()

    if system == "Windows":
        cache_dir = os.path.join(home, "AppData", app_name)
    else:
        cache_dir = os.path.join(home, ".cache", app_name)

    if dataset_name:
        cache_dir = os.path.join(cache_dir, dataset_name)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    return cache_dir
