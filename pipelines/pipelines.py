from nodes.helper import ProgressPipeline

from pipelines import pipeline_registry
from nodes import helper, audio, lexicosyntactic, lexicosyntactic_multi
from utils.segment_mappers import TxtSegments, EafSegments

@pipeline_registry
def split_speech_eaf(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    splitter = EafSegments(in_folder)
    split = audio.SplitSegments("split_speech", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file)

    p = ProgressPipeline(file_finder | split, n_threads=num_threads, quiet=True)

    return p

@pipeline_registry
def split_speech_txt(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    splitter = TxtSegments(in_folder)
    split = audio.SplitSegments("split_speech", out_dir=out_folder, segment_mapping_fn=splitter.get_segs_for_file)

    p = ProgressPipeline(file_finder | split, n_threads=num_threads, quiet=True)

    return p

@pipeline_registry
def opensmile_is10_lld(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = audio.OpenSmileRunner("is10_lld", out_dir=out_folder, conf_file="IS10_paraling.conf", out_flag="-lldcsvoutput")


    p = ProgressPipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p


@pipeline_registry
def opensmile_is10(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = audio.OpenSmileRunner("is10", out_dir=out_folder, conf_file="IS10_paraling.conf", out_flag="-csvoutput")


    p = ProgressPipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p

@pipeline_registry
def matlab(in_folder, out_folder, num_threads):
    from nodes import matlab as mtlb

    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    is10 = mtlb.MatlabRunner("matlab_acoustics", out_dir=out_folder, function="extract_acoustics", out_ext=".txt")

    p = ProgressPipeline(file_finder | is10, n_threads=num_threads, quiet=True)

    return p

@pipeline_registry
def lex(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".txt")

    feats = lexicosyntactic.Lexicosyntactic("lexicosyntactic", out_dir=out_folder, cfg_file="default.conf")

    p = ProgressPipeline(file_finder | feats, n_threads=num_threads, quiet=True)

    return p


@pipeline_registry
def lex_chinese(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".txt")

    feats = lexicosyntactic_multi.ChineseLex("lex_chinese", out_dir=out_folder)

    p = ProgressPipeline(file_finder | feats, n_threads=num_threads, quiet=True, exec_name="ParallelExecutor2")

    return p


@pipeline_registry
def praat_syllable_nuclei(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")

    praat = audio.PraatRunner("praat_syllable_nuclei", out_dir=out_folder)
    p = ProgressPipeline(file_finder | praat, n_threads=num_threads, quiet=True)

    return p


@pipeline_registry
def kaldi_asr(in_folder, out_folder, num_threads):
    file_finder = helper.FindFiles("file_finder", dir=in_folder, ext=".wav")
    resample = audio.ResampleWav("resampled_audio", out_dir=out_folder, new_sr=8000)
    asr = audio.KaldiASR("kaldi_asr", out_dir=out_folder)


    p = ProgressPipeline(file_finder | resample | asr, n_threads=num_threads, quiet=True)

    return p
