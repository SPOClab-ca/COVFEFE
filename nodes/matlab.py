import matlab.engine

import logging

from nodes.helper import FileOutputNode
from utils import file_utils

class MatlabRunner(FileOutputNode):
    def setup(self, function, out_ext, args="-nodesktop -noFigureWindows -nosplash"):
        self.out_ext = out_ext

        self.eng = matlab.engine.start_matlab(args)
        self.eng.addpath("matlab_scripts")
        self.matlab_function = getattr(self.eng, function)

    def run(self, in_file):
        self.log(logging.INFO, "Starting %s" % (in_file))

        out_file = self.derive_new_file_path(in_file, self.out_ext)

        if file_utils.should_run(in_file, out_file):
            try:
                self.matlab_function(in_file, out_file)
            except Exception as e:
                self.log(logging.ERROR, e)

            self.log(logging.INFO, "Done %s -> %s" % (in_file, out_file))

        self.emit(out_file)