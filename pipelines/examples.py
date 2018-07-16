from pyPiper import Pipeline

from pipelines import pipeline_registry
from nodes import helper_nodes, audio_nodes
from utils.segment_mappers import TxtSegments, EafSegments

@pipeline_registry
def split_speech(in_folder, out_folder, num_threads):
    file_finder = helper_nodes.FindFiles("file_finder", dir=in_folder, ext=".wav")

    splitter = EafSegments(in_folder)
    split = audio_nodes.SplitSements("split_speech", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file)

    p = Pipeline(file_finder | split, n_threads=num_threads, quiet=True)

    return p

@pipeline_registry
def opensmile_is10_lld(in_folder, out_folder, num_threads):
    file_finder = helper_nodes.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = audio_nodes.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")


    p = Pipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p