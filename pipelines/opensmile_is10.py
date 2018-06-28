from pyPiper import Pipeline

from pipelines import pipeline_registry
from nodes import helper_nodes, audio_nodes

@pipeline_registry
def opensmile_is10_lld(in_folder, out_folder, num_threads):
    file_finder = helper_nodes.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = audio_nodes.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")


    p = Pipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p