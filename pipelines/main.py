from pyPiper import Pipeline

from pipelines import pipeline_registry
from nodes import helper_nodes, audio_nodes
from utils.segment_mappers import TxtSegments, EafSegments

@pipeline_registry
def main(in_folder, out_folder, num_threads):
    file_finder = helper_nodes.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10_lld = audio_nodes.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf",
                                           out_flag="-lldcsvoutput")
    is10 = audio_nodes.OpenSmileRunner("is10", out_dir=out_folder, conf_file="IS10_paraling.conf",
                                       out_flag="-csvoutput")

    splitter = EafSegments(in_folder)
    split = audio_nodes.SplitSements("utterances", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file,
                                     out_streams=["audio", "label"])

    is10_lld_per_utterance = audio_nodes.OpenSmileRunner("is10_lld_per_utterance", out_dir=out_folder, in_streams="audio",
                                                         conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")
    is10_per_utterance = audio_nodes.OpenSmileRunner("is10_per_utterance", out_dir=out_folder, in_streams="audio",
                                                     conf_file="IS10_paraling.conf", out_flag="-csvoutput")



    p = Pipeline(file_finder | [is10_lld,
                                is10,
                                split | [is10_lld_per_utterance, is10_per_utterance]
                                ],
                 n_threads=num_threads, quiet=True)

    return p