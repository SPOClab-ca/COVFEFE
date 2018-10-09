import logging
import os
import datetime

from nodes.helper import FileOutputNode
from utils import file_utils
from utils.lexicosyntactic import feature
from utils.lexicosyntactic import transcript

import config
from utils.logger import get_logger


class Lexicosyntactic(FileOutputNode):
    # TODO: Figure out and support filler_dir
    def setup(self, cfg_file, utterance_sep=" . ", filler_dir=None):
        self.inited = False
        self.filler_dir = filler_dir
        self.cfg_file = cfg_file
        self.utterance_sep = utterance_sep


    def do_init(self):
        path_output_parses = os.path.join(self.out_dir, "stanford_parses")
        path_output_lu_parses = os.path.join(self.out_dir, "lu_parses")
        path_output_rst = os.path.join(self.out_dir, "rst_output")
        self.output_csv = os.path.join(self.out_dir, 'textfeatures%s.csv' % (
            datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")))

        do_wnic = True

        lexical_list, do_lexical, pragmatic_list, do_pragmatic, semantic_list, do_semantic, syntactic_list, do_syntactic = load_conf(
            self.cfg_file)

        parser_path = config.stanford_parser_path
        pos_tagger_path = config.stanford_pos_path
        lu_analyzer_path = config.lu_analyzer_path
        path_to_stanford_cp = config.path_to_stanford_cp
        cfg_rules_path = config.cfg_rules_path
        path_to_dictionary = config.path_to_dictionary
        path_to_freq_norms = config.path_to_freq_norms
        path_to_image_norms = config.path_to_image_norms
        path_to_anew = config.path_to_anew
        path_to_warringer = config.path_to_warringer
        path_to_mpqa_lexicon = config.path_to_mpqa_lexicon
        path_to_rst_python = config.path_to_rst_python
        path_to_rst = config.path_to_rst
        path_to_lda_model = config.path_to_lda_model
        path_to_lda_wordids = config.path_to_lda_wordids

        self.pos_tagger_path = pos_tagger_path

        self.feature_extractor = feature.FeatureExtractor(
            utterance_sep=self.utterance_sep,
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
            do_lexical=do_lexical,
            do_syntactic=do_syntactic,
            do_semantic=do_semantic,
            do_pragmatic=do_pragmatic,
            lexical_list=lexical_list,
            syntactic_list=syntactic_list,
            semantic_list=semantic_list,
            pragmatic_list=pragmatic_list
        )

        self.filler_files = {os.path.basename(file_utils.strip_ext(x)): x for x in os.listdir(self.filler_dir)}

    def run(self, filepath):
        self.log(logging.INFO, "Starting %s" % (filepath))

        out_file = self.derive_new_file_path(filepath, ".csv")

        if file_utils.should_run(filepath, out_file):
            if not self.inited:
                self.do_init()
                self.inited = True


            try:
                t = transcript.PlaintextTranscript(filepath=filepath, label=None, pos_tagger_path=self.pos_tagger_path)

                transcript_utterances_fillers = None
                if self.filler_dir:
                    file_id = os.path.basename(file_utils.strip_ext(filepath))
                    if file_id in self.filler_files:
                        filler_file = os.path.join(self.filler_dir, self.filler_files[file_id])
                        filler_transcript = transcript.PlaintextTranscript(filepath=filler_file, label=None, pos_tagger_path=self.pos_tagger_path)
                        transcript_utterances_fillers = filler_transcript.tokens

                self.feature_extractor.extract(t, out_csv=out_file, transcript_utterances_fillers=transcript_utterances_fillers)
            except Exception as e:
                self.log(logging.ERROR, "Failed with error %s" % e)

            self.log(logging.INFO, "Done %s -> %s" % (filepath, out_file))

        self.emit(out_file)


def get_features_list(features):
    if len(features) == 1 and features[0] == '':
        return [], False
    features_list = []
    for feature in features:
        feature_name = feature.strip()
        if feature_name == 'all':
            return None, True
        else:
            features_list.append(feature_name)

    return features_list, True


def load_conf(config_file):
    if config_file and os.path.exists(config_file):
        with open(config_file) as f:
            lines = f.readlines()

        if len(lines) < 4:
            get_logger().log(logging.ERROR, 'Error with config file. Using default features.')

        else:
            lexical_features = lines[0].split('#')[0].split(',')
            pragmatic_features = lines[1].split('#')[0].split(',')
            semantic_features = lines[2].split('#')[0].split(',')
            syntactic_features = lines[3].split('#')[0].split(',')

            lexical_list, do_lexical = get_features_list(lexical_features)
            pragmatic_list, do_pragmatic = get_features_list(pragmatic_features)
            semantic_list, do_semantic = get_features_list(semantic_features)
            syntactic_list, do_syntactic = get_features_list(syntactic_features)

            return lexical_list, do_lexical, pragmatic_list, do_pragmatic, semantic_list, do_semantic, syntactic_list, do_syntactic

    return None, True, None, True, None, True, None, True
