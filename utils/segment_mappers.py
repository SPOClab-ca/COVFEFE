import os
from utils.file_utils import strip_ext
from utils.logger import get_logger
from utils.signal_processing import units_to_sample

import logging

class TxtSegments(object):
    def __init__(self, root_dir, ts_units="s", add_extra=False, sep="\t", ext=".txt"):
        self.root_dir = root_dir
        self.ts_units = ts_units
        self.add_extra = add_extra
        self.sep = sep
        self.ext = ext

        self.seg_files = [x for x in os.listdir(self.root_dir) if x.endswith(self.ext)]

    def get_segs_for_file(self, audio_file, sample_rate):
        base_name = strip_ext(os.path.basename(audio_file))

        possible_files = [x for x in self.seg_files if base_name in x]

        res = []

        if len(possible_files) > 0:
            seg_file = os.path.join(self.root_dir, possible_files[0])

            if len(possible_files) > 1:
                get_logger().log(logging.WARNING, "Found multiple matches for %s (%s). Using %s" %
                                 (audio_file, " ".join(possible_files), seg_file))

            with open(seg_file, "r") as f:
                for i, line in enumerate(f):
                    start, end, label = line.strip().split(self.sep)
                    start = units_to_sample(start, self.ts_units, sample_rate)
                    end = units_to_sample(end, self.ts_units, sample_rate)

                    key = "%s-%i-%s-%s" % (base_name, i, str(start), str(end))


                    if self.add_extra:
                        res.append((start, end, key, label))
                    else:
                        res.append((start, end, key))
        else:
            get_logger().log(logging.WARNING, "No seg file found for % in %s" % (audio_file, self.root_dir))

        return res




