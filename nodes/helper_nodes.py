from pyPiper import Node

import os

from utils import file_utils
import nodes

class FileOutputNode(Node):
    def __init__(self, name, out_dir, **kwargs):
        super().__init__(name, **kwargs)
        self.out_dir = os.path.join(out_dir, self.name)
        file_utils.ensure_dir(out_dir)

    def derive_new_file_path(self, old_file, new_ext=None):
        old_fname = os.path.basename(old_file)

        if new_ext is not None:
            new_fname = file_utils.strip_ext(old_fname) + "." + new_ext
        else:
            new_fname = old_fname

        return os.path.join(self.out_dir, new_fname)

    def log(self, level, msg):
        nodes.get_logger().log(level=level, msg="%s: %s" % (self.name, msg))



class FindFiles(Node):
    def setup(self, dir, ext="", prefix=""):
        self.files = file_utils.find_files(dir, prefix=prefix, ext=ext)
        self.stateless = False

    def run(self, data):
        if len(self.files) > 0:
            self.emit(self.files.pop())
        else:
            self.close()