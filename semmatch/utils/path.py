import os


def paths_all_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True