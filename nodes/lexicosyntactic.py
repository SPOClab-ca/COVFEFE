import logging
import os

from nodes.helper import FileOutputNode
from utils import file_utils
from utils.lexicosyntactic import feature
from utils.lexicosyntactic import transcript

import config

class Lexicosyntactic(FileOutputNode):
    # TODO: Figure out and support filler_dir
    def setup(self, cfg_file, filler_dir=None):
        cfg = file_utils.load_json_file(cfg_file)

        utterance_sep = cfg.get("utterance_sep", " . ")
        path_output_lu_parses = cfg.get("path_output_lu_parses", "")
        path_output_parses = cfg.get("path_output_parses", "")
        do_wnic = cfg.get("do_wnic", True)
        path_output_rst = cfg.get("path_output_rst", "")
        do_lexical = cfg.get("do_lexical", True)
        do_syntactic = cfg.get("do_syntactic", True)
        do_semantic = cfg.get("do_semantic", True)
        do_pragmatic = cfg.get("do_pragmatic", True)
        lexical_list = cfg.get("lexical_list", None)
        syntactic_list = cfg.get("syntactic_list", None)
        semantic_list = cfg.get("semantic_list", None)
        pragmatic_list = cfg.get("pragmatic_list", None)

        parser_path = cfg.get("parser_path", config.stanford_parser_path)
        pos_tagger_path = cfg.get("pos_tagger_path", config.stanford_pos_path)
        lu_analyzer_path = cfg.get("lu_analyzer_path", config.lu_analyzer_path)
        path_to_stanford_cp = cfg.get("path_to_stanford_cp", config.path_to_stanford_cp)
        cfg_rules_path = cfg.get("cfg_rules_path", config.cfg_rules_path)
        path_to_dictionary = cfg.get("path_to_dictionary", config.path_to_dictionary)
        path_to_freq_norms = cfg.get("path_to_freq_norms", config.path_to_freq_norms)
        path_to_image_norms = cfg.get("path_to_image_norms", config.path_to_image_norms)
        path_to_anew = cfg.get("path_to_anew", config.path_to_anew)
        path_to_warringer = cfg.get("path_to_warringer", config.path_to_warringer)
        path_to_mpqa_lexicon = cfg.get("path_to_mpqa_lexicon", config.path_to_mpqa_lexicon)
        path_to_rst_python = cfg.get("path_to_rst_python", config.path_to_rst_python)
        path_to_rst = cfg.get("path_to_rst", config.path_to_rst)
        path_to_lda_model = cfg.get("path_to_lda_model", config.path_to_lda_model)
        path_to_lda_wordids = cfg.get("path_to_lda_wordids", config.path_to_lda_wordids)


        self.filler_dir = filler_dir
        self.pos_tagger_path = pos_tagger_path

        self.feature_extractor = feature.FeatureExtractor(
                                                    utterance_sep=utterance_sep,
                                                    path_output_lu_parses=path_output_lu_parses,
                                                    path_output_parses=path_output_parses,
                                                    parser_path=parser_path,
                                                    cfg_rules_path=cfg_rules_path,
                                                    pos_tagger_path=pos_tagger_path,
                                                    path_to_dictionary=path_to_dictionary,
                                                    lu_analyzer_path=lu_analyzer_path,
                                                    path_to_freq_norms=path_to_freq_norms,
                                                    path_to_image_norms=path_to_image_norms,
                                                    path_to_anew=path_to_anew,
                                                    path_to_warringer=path_to_warringer,
                                                    do_wnic=do_wnic,
                                                    path_to_mpqa_lexicon=path_to_mpqa_lexicon,
                                                    path_to_rst_python=path_to_rst_python,
                                                    path_to_rst=path_to_rst,
                                                    path_output_rst=path_output_rst,
                                                    path_to_stanford_cp=path_to_stanford_cp,
                                                    path_to_lda_model=path_to_lda_model,
                                                    path_to_lda_wordids=path_to_lda_wordids,
                                                    do_lexical = do_lexical,
                                                    do_syntactic = do_syntactic,
                                                    do_semantic = do_semantic,
                                                    do_pragmatic = do_pragmatic,
                                                    lexical_list = lexical_list,
                                                    syntactic_list = syntactic_list,
                                                    semantic_list = semantic_list,
                                                    pragmatic_list = pragmatic_list
        )

        self.filler_files = {os.path.basename(file_utils.strip_ext(x)): x for x in os.listdir(self.filler_dir)}

    def run(self, filepath):
        self.log(logging.INFO, "Starting %s" % (filepath))

        out_file = self.derive_new_file_path(filepath, ".csv")

        if file_utils.should_run(filepath, out_file):

            t = transcript.PlaintextTranscript(filepath=filepath, label=None, pos_tagger_path=self.pos_tagger_path)

            transcript_utterances_fillers = None
            if self.filler_dir:
                file_id = os.path.basename(file_utils.strip_ext(filepath))
                if file_id in self.filler_files:
                    filler_file = os.path.join(self.filler_dir, self.filler_files[file_id])
                    filler_transcript = transcript.PlaintextTranscript(filepath=filler_file, label=None, pos_tagger_path=self.pos_tagger_path)
                    transcript_utterances_fillers = filler_transcript.tokens

            self.feature_extractor.extract(t, out_csv=out_file, transcript_utterances_fillers=transcript_utterances_fillers)

            self.log(logging.INFO, "Done %s -> %s" % (filepath, out_file))

        self.emit(out_file)
