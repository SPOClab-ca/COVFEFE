import os
from utils.file_utils import strip_ext
from utils.logger import get_logger
from utils.signal_processing import units_to_sample
from utils.eaf_helper import eaf2df

import logging

class TxtSegments(object):
    def __init__(self, root_dir, ts_units="s", add_labels=False, sep="\t", ext=".txt"):
        self.root_dir = root_dir
        self.ts_units = ts_units
        self.add_labels = add_labels
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

            res = get_txt_segs(seg_file, sample_rate, base_name, self.add_labels, self.sep, self.ts_units)
        else:
            get_logger().log(logging.WARNING, "No seg file found for %s in %s" % (audio_file, self.root_dir))

        return res


class EafSegments(object):
    def __init__(self, root_dir, ts_units="s", add_labels=False, ext=".eaf"):
        self.root_dir = root_dir
        self.ts_units = ts_units
        self.add_labels = add_labels
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

            res = get_eaf_segs(seg_file, sample_rate, base_name, self.add_labels)

        else:
            get_logger().log(logging.WARNING, "No seg file found for %s in %s" % (audio_file, self.root_dir))

        return res



def get_eaf_segs(seg_file, sample_rate, base_name, add_labels):
    df = eaf2df(seg_file)

    df["start"] = df["timeslot_start_ms"] * 1000 * sample_rate
    df["end"] = df["timeslot_end_ms"] * 1000 * sample_rate
    df["key"] = df.apply(lambda x: "%s-%s-%i-%i" % (base_name, x["annotation_id"], x["start"], x["end"]), axis=1)

    if add_labels:
        cols = ["start", "end", "key", "annotation"]
    else:
        cols = ["start", "end", "key"]

    res = list(df[cols].to_records(index=False))

    return res


def get_txt_segs(seg_file, sample_rate, base_name, add_labels, sep, ts_units):
    res = []
    with open(seg_file, "r") as f:
        for i, line in enumerate(f):
            start, end, label = line.strip().split(sep)
            start = units_to_sample(start, ts_units, sample_rate)
            end = units_to_sample(end, ts_units, sample_rate)

            key = "%s-%i-%s-%s" % (base_name, i, str(start), str(end))

            if add_labels:
                res.append((start, end, key, label))
            else:
                res.append((start, end, key))

    return res