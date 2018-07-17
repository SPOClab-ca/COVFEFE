import os
import json


def should_run(in_file, out_file):
    if not os.path.isfile(out_file):
        return True

    in_modified = os.path.getmtime(in_file)
    out_modified = os.path.getmtime(out_file)

    return in_modified > out_modified


def ensure_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def load_json_file(fname):
    with open(fname) as jsonFile:
        data = json.load(jsonFile)
        return data


def write_json_file(f, data):
    with open(f, 'w') as jsonFile:
        json.dump(data, jsonFile)


def strip_ext(fname):
    return fname.rsplit(".", 1)[0]


def find_files(path, ext="", prefix=""):
    return [os.path.join(path, x) for x in os.listdir(path) if x.endswith(ext) and x.startswith(prefix)]


def locate_file(file_name, possible_paths=[], use_path=False):
    # Is file_name an absolute path?
    if os.path.isfile(file_name):
        return file_name

    if use_path:
        possible_paths += os.environ["PATH"].split(":")

    for possible_path in possible_paths:
        p = os.path.join(possible_path, file_name)
        if os.path.isfile(p):
            return p

    cwd = os.getcwd()
    raise ValueError("File %s does not exist in any of [%s]" % (file_name, ", ".join([cwd] + possible_paths)))