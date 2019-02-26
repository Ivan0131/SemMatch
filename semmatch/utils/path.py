import os


def paths_all_exist(paths):
    for path in paths:
        if not os.path.exists(path):
            return False
    return True


def get_file_extension(path: str, dot=True, lower: bool = True):
    ext = os.path.splitext(path)[1]
    ext = ext if dot else ext[1:]
    return ext.lower() if lower else ext


