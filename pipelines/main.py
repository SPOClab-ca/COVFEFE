from nodes.helper import ProgressPipeline

from pipelines import pipeline_registry
from nodes import helper, audio
from utils.segment_mappers import TxtSegments, EafSegments

@pipeline_registry
def main(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    resample = audio.ResampleWav("resampled_audio", out_dir=out_folder, new_sr=8000)

    is10_lld = audio.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf",
                                     out_flag="-lldcsvoutput")
    is10 = audio.OpenSmileRunner("is10", out_dir=out_folder, conf_file="IS10_paraling.conf",
                                 out_flag="-csvoutput")

    splitter = EafSegments(in_folder)
    split = audio.SplitSegments("utterances", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file,
                               out_streams=["audio", "label"])

    is10_lld_per_utterance = audio.OpenSmileRunner("is10_lld_per_utterance", out_dir=out_folder, in_streams="audio",
                                                   conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")
    is10_per_utterance = audio.OpenSmileRunner("is10_per_utterance", out_dir=out_folder, in_streams="audio",
                                               conf_file="IS10_paraling.conf", out_flag="-csvoutput")

    asr = audio.KaldiASR("kaldi_asr", out_dir=out_folder, in_streams="audio")


    p = ProgressPipeline(file_finder | resample | [is10_lld,
                                is10,
                                split | [asr, is10_lld_per_utterance, is10_per_utterance]
                                ],
                 n_threads=num_threads, quiet=True)

    return p