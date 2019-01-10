import subprocess
import collections
import csv
import os
import re
import logging
import statistics
import wordfreq
import nltk.tree

from nodes.helper import FileOutputNode
from utils import file_utils
import config


SENTENCE_TOKENS = '.。!?！？'

POS_TAGS = [
    "AD","AS","BA","CC","CD","CS","DEC","DEG","DER","DEV","DT","ETC","FW","IJ",
    "JJ","LB","LC","M","MSP","NN","NR","NT","OD","ON","P","PN","PU","SB","SP",
    "VA","VC","VE","VV","X","XX","URL"
]


class MultilangTranscript(object):
    def __init__(self, filepath, out_file, output_parse_dir, cfg_rules):
        self.filepath = filepath
        self.out_file = out_file
        self.output_parse_dir = output_parse_dir
        self.cfg_rules = cfg_rules

        self.features = collections.OrderedDict()
        self.pos_tags = []
        self.parse_trees = []

    def _ratio(self, a, b):
        """Divide but default to 1 if denominator is zero"""
        if b == 0:
            return 1
        else:
            return a / b


    def _run_chinese_corenlp(self, filepath):
        self.corenlp_out_file = os.path.join(self.output_parse_dir, os.path.basename(filepath) + '.out')

        if not os.path.isfile(self.corenlp_out_file):
            # lexparser_chinese.sh [output_dir] [transcript_file]
            subprocess.call([
                os.path.join(config.path_to_stanford_cp, 'lexparser_chinese.sh'),
                self.output_parse_dir,
                filepath
            ])

    def _parse_corenlp_output(self):
        with open(self.corenlp_out_file) as f:
            for line in f.readlines():
                line = line[:-1]

                match = re.search(r'PartOfSpeech=([A-Z]+)\]', line)
                if match:
                    tag = match.group(1)
                    assert(tag in POS_TAGS)
                    self.pos_tags.append(tag)

        # Count POS tag features
        for pos_tag in POS_TAGS:
            count = 0
            for tag in self.pos_tags:
                if tag == pos_tag:
                    count += 1
            self.features['pos_' + pos_tag] = count
            self.features['pos_ratio_' + pos_tag] = self._ratio(count, len(self.pos_tags))

        # A few special ones
        self.features['ratio_pronoun_noun'] = self._ratio(self.features['pos_PN'], (self.features['pos_PN'] + self.features['pos_NN']))
        self.features['ratio_noun_verb'] = self._ratio(self.features['pos_NN'], (self.features['pos_NN'] + self.features['pos_VV']))

        self.features['num_tokens'] = len(self.pos_tags)


        # Parse constituency trees
        with open(self.corenlp_out_file) as f:

            partial_parse_tree = ''
            for line in f.readlines():

                # If it starts with '(', then begin a new tree
                if line.startswith('('):
                    if len(partial_parse_tree) > 0:
                        try:
                            parse_tree = nltk.tree.Tree.fromstring(partial_parse_tree)
                            self.parse_trees.append(parse_tree)
                        except:
                            pass
                        partial_parse_tree = ''

                line = line.strip()
                if line.startswith('('):
                    partial_parse_tree += ' ' + line

            # Last parse tree
            try:
                parse_tree = nltk.tree.Tree.fromstring(partial_parse_tree)
                self.parse_trees.append(parse_tree)
            except:
                pass

        # Parse tree features
        tree_heights = []
        for tree in self.parse_trees:
            tree_heights.append(tree.height())
        self.features['max_tree_height'] = max(tree_heights)
        self.features['mean_tree_height'] = statistics.mean(tree_heights)
        self.features['median_tree_height'] = statistics.median(tree_heights)

        # Count CFG rules
        num_cfg_productions = 0
        dtree = collections.defaultdict(int)
        for tree in self.parse_trees:
            for cfg_rule in tree.productions():
                if cfg_rule.is_nonlexical():
                  cfg_rule_str = str(cfg_rule).replace(' ', '_')
                  dtree[cfg_rule_str] += 1
                  num_cfg_productions += 1

        for cfg_rule in self.cfg_rules:
            self.features[cfg_rule] = dtree[cfg_rule] / num_cfg_productions


    def compute_word_frequency_norms(self):
        freqs = []
        for char in self.tokens:
            freq = wordfreq.word_frequency(char, 'zh')

            if freq == 0:
                continue

            freqs.append(freq)

        try:
            self.features['mean_word_frequency'] = statistics.mean(freqs)
            self.features['median_word_frequency'] = statistics.median(freqs)
        except:
            self.features['mean_word_frequency'] = 0
            self.features['median_word_frequency'] = 0


    def write_features(self, out_file, debug):
        if debug:
            for k, v in self.features.items():
                print(k, v)
        else:
            with open(out_file, 'w') as f:
                csvw = csv.writer(f)
                csvw.writerow(list(self.features.keys()))
                csvw.writerow(list(self.features.values()))

    def _calc_ttr(self, text):
        """TTR = unique words / all words"""
        N = len(text)
        V = len(set(text))
        return self._ratio(V, N)


    def compute_basic_word_stats(self):
        num_sentences = len([x for x in self.tokens if x in SENTENCE_TOKENS])
        num_words = len(self.tokens) - num_sentences
        ttr = self._calc_ttr([x for x in self.tokens if x not in SENTENCE_TOKENS])
        word_lengths = [len(x) for x in self.tokens if x not in SENTENCE_TOKENS]

        self.features['num_sentences'] = num_sentences
        self.features['mean_words_per_sentence'] = self._ratio(num_words, num_sentences)
        self.features['ttr'] = ttr

    def run(self):
        if file_utils.should_run(self.filepath, self.out_file):
            self.features['FileID'] = self.filepath

            with open(self.filepath) as f:
                self.tokens = f.read()

            self.compute_basic_word_stats()
            self.compute_word_frequency_norms()

            self._run_chinese_corenlp(self.filepath)
            self._parse_corenlp_output()
            self.write_features(self.out_file, debug=False)


class MultilingualLex(FileOutputNode):
    def setup(self):
        self.output_parse_dir = os.path.join(self.out_dir, "stanford_parses")
        with open('/h/bai/research/bai-alzheimer/naacl19/top_chinese_cfg.txt') as cfgf:
            self.cfg_rules = list(map(lambda x: x[:-1], cfgf.readlines()))

    def run(self, filepath):
        self.log(logging.INFO, "Starting %s" % (filepath))
        out_file = self.derive_new_file_path(filepath, ".csv")

        transcript = MultilangTranscript(filepath, out_file, self.output_parse_dir, self.cfg_rules)
        try:
            transcript.run()
        except:
            print('Failed:', filepath)

        self.emit(out_file)
