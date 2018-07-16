from pyPiper import Pipeline

from pipelines import pipeline_registry
from nodes import helper_nodes, audio_nodes
from utils.segment_mappers import TxtSegments

@pipeline_registry
def split_speech(in_folder, out_folder, num_threads):
    file_finder = helper_nodes.FindFiles("file_finder", dir=in_folder, ext=".wav")

    splitter = TxtSegments(in_folder)
    split = audio_nodes.SplitSements("split_speech", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file)

    p = Pipeline(file_finder | split, n_threads=num_threads, quiet=True)

    return p