
import collections
import csv
import glob
import os
import re

import nltk.probability
import nltk.tree
from nltk.corpus import wordnet_ic as wnic
from nltk.corpus import cmudict

from utils.lexicosyntactic import lexical_features
from utils.lexicosyntactic import pragmatic_features
from utils.lexicosyntactic import semantic_features
from utils.lexicosyntactic import syntactic_features

from utils.lexicosyntactic import functions
from utils.lexicosyntactic import transcript
from utils.lexicosyntactic import yngve

from utils import file_utils
import config

class Feature(object):

    def __init__(self, feature_type, name, value):
        '''Parameters:
        feature_type : string. Used to group features. E.g., "lexical", "syntactic", "semantic"
        name : string, the name of the feature. E.g., "avg_cosine_distance"
        value : float, the value of the feature. E.g., "1.0"
        '''
        self.feature_type = feature_type
        self.name = name
        self.value = value

class FeatureSet(object):

    def __init__(self, features=[]):
        '''Parameters:
        features : optional, list. A list of Feature objects.
        '''
        self.features = features

    def add(self, new_feature):
        if type(new_feature) == list:
            self.features += new_feature
        else:
            self.features += [new_feature]

    def get_length(self):
        '''Return the number of features in the set.'''
        return len(self.features)

    def __getitem__(self, index):
        '''Overload the [] operator to enable subscription.'''
        return self.features[index]

    def __str__(self):
        return "FeatureSet(%d features)" % (self.get_length())

    def __repr__(self):
        return self.__str__()


class FeatureExtractor(object):

    # Remove stopwords (don't use the NLTK stopword list because it is too extensive and contains some verbs)
    stopwords = ['the', 'and', 'is', 'a', 'to', 'i', 'on', 'in', 'of', 'it'] # top 10, function words only
    transcript_extension = 'txt'

    # Initialize lexical lemmatizer
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    # Define variables needed for feature extraction
    inflected_verb_tags = ['VBD', 'VBG', 'VBN', 'VBZ']
    light_verbs = ["be", "have", "come", "go", "give", "take", "make", "do", "get", "move", "put"]
    demonstratives = ["this", "that", "these", "those"]
    function_tags = ["DT", "PRP", "PRP$", "WDT", "WP", "WP$", "CC", "RP", "MD", "IN"]
    # From Szmrecsanyi 2004
    subordinate = ["because", "since", "as", "when", "that", "although", "though", "while", "before", "after", "even", "now",
                   "once", "than", "unless", "until", "when", "whenever", "where", "while", "who", "whoever", "why"]

    # When checking for English words, the following tokens should be ignored:
    word_exceptions = ["n't", "'m", "'s", "'ll", "'re", "'d", "'ve"]

    # Value to use for infinity
    inf_value = 10^10
    nan_value = -1

    def __init__(self, utterance_sep, path_output_lu_parses, path_output_parses,
                 parser_path, cfg_rules_path, pos_tagger_path=None, path_to_freq_norms=None, path_to_image_norms=None,
                 path_to_dictionary=None, lu_analyzer_path=None, path_to_anew=None, path_to_warringer=None, do_wnic=False,
                 path_to_rst_python=None, path_to_rst=None, path_output_rst=None, path_to_stanford_cp=None,
                 path_to_mpqa_lexicon=None, path_to_lda_model=None, path_to_lda_wordids=None, do_lexical=True,
                 do_syntactic=True, do_semantic=True, do_pragmatic=False, lexical_list=None, syntactic_list=None,
                 semantic_list=None, pragmatic_list=None):
        '''Parameters:
        source_transcript : list of strings. Full paths to directories containing transcripts (with no filler annotations)
        source_transcript_fillers : list of string. Full paths to a directories containing transcripts with filler annotations
        utterance_sep : string. The string that delimits utterance boundaries in the transcript
        path_lu_output_parses : string. The absolute path to a directory that will store the Lu features and parses.
        path_output_parses : string. The absolute path to a directory that will store the parse trees produced for the data.
        parser_path : string. The absolute path to a directory containing a Stanford lexparser
        cfg_rules_path : string. The absolute path to a file containing cfg productions to be extracted (one per line)
        path_output_lda_topics: string. The absolute path to the csv file where key-value topics will be stored.
        pos_tagger_path : optional, string. Full path to a directory containing a Stanford POS tagger
        path_to_freq_norms : optional, string. Full path to a file containing frequency norms
        path_to_image_norms : optional, string. Full path to a file containing imageability norms
        path_to_dictionary : optional, string. Full path to a file containing valid words for the language
        lu_analyzer_path : optional
        path_to_rst_python : optional, string. Full path to virtualenv python, for RST
        path_to_rst : optional, string. Full path to folder with RST's 'parse.py'
        path_output_rst: optional, string. Full path to where RST stores its results
        path_to_lda_model : string. Full path to trained LDA model.
        path_to_lda_wordids : string. Full path to word IDs used in trained LDA model.
        '''

        self.utterance_sep = utterance_sep

        self.output_rst_dir = os.path.abspath(path_output_rst)
        self.output_parse_dir = os.path.abspath(path_output_parses)
        self.output_lu_parse_dir = os.path.abspath(path_output_lu_parses)

        self.pos_tagger_path = pos_tagger_path
        self.parser_path = parser_path
        self.cfg_rules_path = cfg_rules_path
        self.path_to_mpqa_lexicon = path_to_mpqa_lexicon
        self.path_to_rst_python = path_to_rst_python
        self.path_to_rst = path_to_rst
        self.path_to_stanford_cp = path_to_stanford_cp
        self.path_to_lda_model = path_to_lda_model
        self.path_to_lda_wordids = path_to_lda_wordids

        self.do_lexical = do_lexical
        self.do_syntactic = do_syntactic
        self.do_semantic = do_semantic
        self.do_pragmatic = do_pragmatic
        self.lexical_list = lexical_list
        self.syntactic_list = syntactic_list
        self.semantic_list = semantic_list
        self.pragmatic_list = pragmatic_list

        file_utils.ensure_dir(self.output_parse_dir)
        file_utils.ensure_dir(self.output_lu_parse_dir)
        file_utils.ensure_dir(self.output_rst_dir)

        # self.transcript_set = transcript.TranscriptSet(dataset=[])

        # Get lexical norms
        if path_to_freq_norms is not None:
            self.norms_freq = functions.get_frequency_norms(path_to_freq_norms)
        else: # default
            self.norms_freq = functions.get_frequency_norms()

        if path_to_image_norms is not None:
            self.norms_image = functions.get_imageability_norms(path_to_image_norms)
        else: # default
            self.norms_image = functions.get_imageability_norms()

        if path_to_anew is not None:
            self.norms_anew = functions.get_anew_norms(path_to_anew)
        else: # default
            self.norms_anew = None

        # Warringer
        if path_to_warringer is not None:
            self.norms_warringer = functions.get_warringer_norms(path_to_warringer)
        else: # default
            self.norms_warringer = functions.get_warringer_norms()

        # MPQA
        if path_to_mpqa_lexicon is not None:
            [self.mpqa_words, self.mpqa_types, self.mpqa_polarities] = functions.get_mpqa_lexicon(path_to_mpqa_lexicon)
        else: # default
            [self.mpqa_words, self.mpqa_types, self.mpqa_polarities] = functions.get_mpqa_lexicon()

        # Set up the dictionary of valid words for the language
        if path_to_dictionary is not None:
            source_dict = path_to_dictionary
        else:
            source_dict = os.path.abspath("../feature_extraction/text/american-english") # default
        with open(source_dict, 'r') as fin_dict:
            words = fin_dict.readlines()
            self.dictionary_words = set(word.strip().lower() for word in words)
        self.prondict = cmudict.dict()

        if lu_analyzer_path is not None:
            self.lu_analyzer_path = lu_analyzer_path
        else:
            self.lu_analyzer_path = os.path.abspath('../L2SCA-2011-10-10/')

        # semantics
        if do_wnic:
            self.brown_ic = wnic.ic('ic-brown.dat')      # FR: it would be nice to have a dat based on normative data, baby
            self.semcor_ic = wnic.ic('ic-semcor.dat')
        else:
            self.brown_ic = []
            self.semcor_ic = []


    def extract(self, t, out_csv, transcript_utterances_fillers=None):
        '''Parameters:
        debug_output : optional, string. Full path to a directory to store partial results. If None,
                       no partial results are written out.
        do_lexical : optional (default=True), Boolean. If True, extract lexical features.
        do_syntactic : optional (default=True), Boolean. If True, extract syntactic features.
        lexical_list : optional (default=None), list or None. If not None, limit extracted lexical features to given list.
        syntactic_list : optional (default=None), list or None. If not None, limit extracted syntactic features to given list.

        Return: nothing.'''

        # Get a list of lists of tokens (each row is an utterance)
        transcript_utterances = t.tokens
        pos_utterances = t.get_pos_tagged()

        total_words = len([token_tuple for utt in pos_utterances for token_tuple in utt])

        # Create a list of Feature objects, and add to transcript. Always sort in ascending order of
        # feature name, so we get the same order for all transcripts.
        features = []

        # LEXICAL FEATURES (get a dict of key=feat_name, value=feat_value).
        # Then, normalize by overall transcript word counts and obtain ratios.
        if self.do_lexical:
            features_lexical, sorted_lexical_names = self.extract_lexical(transcript_utterances, transcript_utterances_fillers, pos_utterances, total_words, list_features=self.lexical_list)
            features_lexical = self.normalize_lexical_features(features_lexical, total_words)
            for feat_name in sorted_lexical_names:
                features += [Feature(feature_type="lexical", name=feat_name, value=features_lexical[feat_name])]

        # SYNTACTIC FEATURES (get a dict of key=feat_name, value=feat_value)
        if self.do_syntactic:
            features_syntactic, sorted_syntactic_names = self.extract_syntactic(t.filepath, t.filename, transcript_utterances, self.syntactic_list)
            features_syntactic = self.normalize_syntactic_features(features_syntactic, total_words)
            for feat_name in sorted_syntactic_names:
                features += [Feature(feature_type="syntactic", name=feat_name, value=features_syntactic[feat_name])]

        # SEMANTIC FEATURES (get a dict of key=feat_name, value=feat_value)
        if self.do_semantic:
            features_semantic, sorted_semantic_names = self.extract_semantic(t.filepath, t.filename, transcript_utterances, self.semantic_list)
            # features_semantic = self.normalize_semantic_features(features_semantic, total_words) # TODO
            for feat_name in sorted_semantic_names:
                features += [Feature(feature_type="semantic", name=feat_name, value=features_semantic[feat_name])]

        # PRAGMATIC FEATURES (get a dict of key=feat_name, value=feat_value)
        if self.do_pragmatic:
            features_pragmatic, sorted_pragmatic_names = self.extract_pragmatic(t.filepath, t.filename, transcript_utterances, total_words, self.pragmatic_list)
            # features_pragmatic = self.normalize_pragmatic_features(features_pragmatic, total_words) # TODO?
            for feat_name in sorted_pragmatic_names:
                features += [Feature(feature_type="pragmatic", name=feat_name, value=features_pragmatic[feat_name])]

        # Add all extracted features to transcript
        t.add_feature(features)

        # self.transcript_set.append(t)

        # Assume all transcripts in the set have the same features in the same order
        headers = [feat.name for feat in t.feature_set]
        with open(out_csv, 'w') as csvfout:
            csvwriter = csv.writer(csvfout, delimiter=',', quoting=csv.QUOTE_ALL)
            csvwriter.writerow(headers)
            csvwriter.writerow([feat.value for feat in t.feature_set])


    def extract_lexical(self, transcript_utterances, transcript_utterances_fillers, pos_utterances, total_words, list_features=None):
        '''Parameters:
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        transcript_utterances_fillers : list of lists of strings (words); each row is a plaintext utterance containing filled pauses
        pos_utterances : list of lists of tuples of (token, POStag); each row is an utterance. No filled pauses.
        total_words : int, the total number of words in the transcript (used as normalization constant).
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
                        Possible list_features options: wordnet, cosine_distance, fillers, vocab_richness, mpqa, readability,
                        stanford_sentiment, pos_counts, pos_ratios, freq_norms, image_norms, anew_norms, warringer_norms, density

        Return:
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''
        feature_dict = collections.defaultdict(int)
        sorted_keys = []

        # Filter out any punctuation tags
        regex_pos_content = re.compile(r'^[a-zA-Z$]+$')
        pos_tokens = [] # list of tuples of (token, POStag) of non-punctuation tokens
        lemmatized_tokens = [] # list of lemmatized non-punctuation tokens
        for utt in pos_utterances:
            for token_tuple in utt:
                # If the POS tag is not that of punctuation, add to tokens
                if regex_pos_content.findall(token_tuple[1]):
                    pos_tokens += [token_tuple]

                    feature_dict['word_length'] += len(token_tuple[0])

                    if token_tuple[0] not in self.dictionary_words and token_tuple[0] not in self.word_exceptions:
                        feature_dict['NID'] += 1

                    # Lemmatize according to the type of the word
                    lemmatized_token = self.lemmatizer.lemmatize(token_tuple[0], functions.pos_treebank2wordnet(token_tuple[1]))
                    lemmatized_tokens += [lemmatized_token]

        if list_features is None or 'wordnet' in list_features:

            wordnet_keys, wordnet_features = lexical_features.get_wordnet_features(pos_utterances,
                                                                                   self.nan_value)

            sorted_keys += wordnet_keys
            for feature in wordnet_keys:
                feature_dict[feature] = wordnet_features[feature]

        if list_features is None or 'cosine_distance' in list_features:

            cosine_keys, cosine_features = lexical_features.get_cosine_distance(transcript_utterances,
                                                                                self.stopwords,
                                                                                self.inf_value)
            sorted_keys += cosine_keys
            for feature in cosine_keys:
                feature_dict[feature] = cosine_features[feature]

        if list_features is None or 'fillers' in list_features:

            filler_keys, filler_features = lexical_features.get_filler_counts(transcript_utterances_fillers)
            sorted_keys += filler_keys
            for feature in filler_keys:
                feature_dict[feature] = filler_features[feature]

        if list_features is None or 'vocab_richness' in list_features:

            vocab_richness_keys, vocab_richness_features = lexical_features.get_vocab_richness_measures(pos_tokens,
                                                                                                        lemmatized_tokens,
                                                                                                        total_words,
                                                                                                        self.inf_value)
            sorted_keys += vocab_richness_keys
            for feature in vocab_richness_keys:
                feature_dict[feature] = vocab_richness_features[feature]

        if list_features is None or 'mpqa' in list_features:

            mpqa_keys, mpqa_features = lexical_features.get_mpqa_norm_features(lemmatized_tokens,
                                                                               self.mpqa_words,
                                                                               self.mpqa_types,
                                                                               self.mpqa_polarities,
                                                                               self.nan_value)
            sorted_keys += mpqa_keys
            for feature in mpqa_keys:
                feature_dict[feature] = mpqa_features[feature]

        if list_features is None or 'readability' in list_features:
            readbility_keys, readbility_features = lexical_features.get_readability_measures(transcript_utterances,
                                                                                             self.prondict,
                                                                                             total_words,
                                                                                             self.nan_value)
            sorted_keys += readbility_keys
            for feature in readbility_keys:
                feature_dict[feature] = readbility_features[feature]

        if list_features is None or 'liwc' in list_features:
            liwc_keys, liwc_features = lexical_features.get_liwc_features(transcript_utterances, self.nan_value)
            sorted_keys += liwc_keys
            for feature in liwc_keys:
                feature_dict[feature] = liwc_features[feature]

        if list_features is None or 'stanford_sentiment' in list_features:
            stanford_keys, stanford_features = lexical_features.get_stanford_sentiment_features(transcript_utterances,
                                                                                                self.path_to_stanford_cp,
                                                                                                self.nan_value)
            sorted_keys += stanford_keys
            for feature in stanford_keys:
                feature_dict[feature] = stanford_features[feature]

        if list_features is None or ('pos_counts' in list_features or 'pos_ratios' in list_features
                                     or 'freq_norms' in list_features or 'image_norms' in list_features
                                     or 'anew_norms' in list_features or 'warringer_norms' in list_features
                                     or 'density' in list_features):

            get_pos_counts = True if list_features is None or 'pos_counts' in list_features else False
            get_pos_ratios = True if list_features is None or 'pos_ratios' in list_features else False
            get_frequency_norms = True if list_features is None or 'freq_norms' in list_features else False
            get_image_norms = True if list_features is None or 'image_norms' in list_features else False
            get_warringer_norms = True if list_features is None or 'warringer_norms' in list_features else False
            get_density = True if list_features is None or 'density' in list_features else False

            if config.path_to_anew:
                get_anew_norms = True if list_features is None or 'anew_norms' in list_features else False
            else:
                get_anew_norms = False

            pos_keys, pos_features = lexical_features.get_pos_features(pos_utterances, total_words, self.lemmatizer,
                                                                       self.norms_freq, self.norms_image, self.norms_anew, self.norms_warringer,
                                                                       self.function_tags, self.inflected_verb_tags, self.light_verbs,
                                                                       self.subordinate, self.demonstratives, self.dictionary_words, self.word_exceptions,
                                                                       self.inf_value, self.nan_value,
                                                                       get_pos_counts=get_pos_counts,
                                                                       get_pos_ratios=get_pos_ratios,
                                                                       get_frequency_norms=get_frequency_norms,
                                                                       get_image_norms=get_image_norms,
                                                                       get_anew_norms=get_anew_norms,
                                                                       get_warringer_norms=get_warringer_norms,
                                                                       get_density=get_density)
            sorted_keys += pos_keys
            for feature in pos_keys:
                feature_dict[feature] = pos_features[feature]

        return feature_dict, sorted_keys


    def extract_syntactic(self, transcript_filepath, transcript_filename, transcript_utterances, list_features=None):
        '''Parameters:
        transcript_filepath : string. The absolute path to the file containing the transcript.
        transcript_filename : string. The filename of the file containing the transcript.
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
                        Possible features: lu_complexity, parsetrees,

        Return:
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''
        feature_dict = {}
        sorted_keys = []

        if list_features is None or 'lu_complexity' in list_features:

            lu_keys, lu_features = syntactic_features.get_lu_complexity_features(self.lu_analyzer_path,
                                                                                 transcript_filepath,
                                                                                 transcript_filename,
                                                                                 self.output_lu_parse_dir)
            sorted_keys += lu_keys
            for feature in lu_features:
                feature_dict[feature] = lu_features[feature]

        if list_features is None or 'parsetrees' in list_features:

            parsetree_keys, parsetree_features = syntactic_features.get_parsetree_features(self.parser_path,
                                                                                           self.cfg_rules_path,
                                                                                           transcript_filepath,
                                                                                           transcript_filename,
                                                                                           self.output_parse_dir)
            sorted_keys += parsetree_keys
            for feature in parsetree_features:
                feature_dict[feature] = parsetree_features[feature]

        return feature_dict, sorted_keys

    def extract_semantic(self, transcript_filepath, transcript_filename, transcript_utterances, list_features=None):
        '''Parameters:
        transcript_filepath : string. The absolute path to the file containing the transcript.
        transcript_filename : string. The filename of the file containing the transcript.
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
                        Possible features: wordnet
        Return:
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''
        feature_dict = {}
        sorted_keys = []

        if list_features is None or 'wordnet' in list_features:
            wordnet_keys, wordnet_features = semantic_features.get_wordnet_features(transcript_utterances,
                                                                                    brown_information_content=self.brown_ic,
                                                                                    semcor_information_content=self.semcor_ic,
                                                                                    nan_value=self.nan_value)

            sorted_keys += wordnet_keys
            for feature in wordnet_features:
                feature_dict[feature] = wordnet_features[feature]

        return feature_dict, sorted_keys

    def extract_pragmatic(self, transcript_filepath, transcript_filename, transcript_utterances, total_words, list_features=None):
        '''Parameters:
        transcript_filepath : string. The absolute path to the file containing the transcript.
        transcript_filename : string. The filename of the file containing the transcript.
        transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
        list_features : optional (default=None), list or None. If not None, limit the extracted features to those in list.
                        Possible features: lda, rst
        Return:
        feature_dict : dict of key=feat_name, value=feat_value.
        sorted_keys : list of strings (feature names).
        '''

        feature_dict = {}
        sorted_keys = []

        if list_features is None or 'lda' in list_features:
            lda_keys = ['kurtosis_lda', 'skewness_lda', 'entropy_lda']

            topic_probabilities, kurtosis, skewness, entropy = pragmatic_features.get_lda_topics(transcript_utterances,
                                                                                                 self.path_to_lda_model, self.path_to_lda_wordids)

            feature_dict['kurtosis_lda'] = kurtosis
            feature_dict['skewness_lda'] = skewness
            feature_dict['entropy_lda'] = entropy

            for idx, topic_probability in enumerate(topic_probabilities):
                topic_key = 'topic%d' % idx
                lda_keys += [topic_key]
                feature_dict[topic_key] = topic_probability

            sorted_keys += lda_keys

        if list_features is None or 'rst' in list_features:
            rst_keys = ['rst_num_attribution', 'rst_num_background', 'rst_num_cause', 'rst_num_comparison', 'rst_num_condition',
                        'rst_num_contrast', 'rst_num_elaboration', 'rst_num_enablement', 'rst_num_evaluation',
                        'rst_num_explanation', 'rst_num_joint', 'rst_num_manner-means', 'rst_num_sameUnit',
                        'rst_num_summary', 'rst_num_temporal', 'rst_topic-comment']
            sorted_keys += rst_keys
            RSTcounts = pragmatic_features.get_rstHistogram(transcript_filepath, transcript_filename,
                                                            self.path_to_rst_python, self.path_to_rst, self.output_rst_dir,
                                                            self.nan_value)

            for k in range(len(rst_keys)):
                # Normalize counts by number of words (FR: is this what we want?)
                feature_dict[rst_keys[k]] = self.nan_value if total_words == 0 else 1.0*RSTcounts[k]/total_words

        return feature_dict, sorted_keys

    def normalize_syntactic_features(self, feature_dict, normalization_factor):

        # NORMALIZATION by total number of words
        norm_by_total_words = ["T", "C", "CN", "CP", "CT", "VP", "DC"]

        if normalization_factor > 0:
            for feat_name in norm_by_total_words:
                if feat_name in feature_dict:
                    feature_dict[feat_name] = feature_dict[feat_name] * 1.0 / normalization_factor
        return feature_dict

    def normalize_lexical_features(self, feature_dict, normalization_factor):

        # NORMALIZATION of pos counts
        # feature IDs that should be normalized by the total number of words in the transcript
        norm_by_total_words = ["nouns", "verbs", "pronouns", "word_length", "function", "demonstratives",
                               "prepositions", "adverbs", "adjectives",
                               "determiners", "coordinate", "subordinate", "NID", "um", "uh", "fillers",
                               "T", "C", "CN", "CP", "CT", "VP", "DC"]

        if normalization_factor > 0:

            # Normalize all features by the total number of words in transcript
            for feat_name in norm_by_total_words:
                if feat_name in feature_dict:
                    feature_dict[feat_name] = 1.0 * feature_dict[feat_name] / normalization_factor

        return feature_dict

    def write_to_csv(self, output_csv):
        '''Parameters:
        output_csv : string. The full path and filename of the output CSV where we store the features.
                     The first line contains the feature names, and every line thereafter corresponds to
                     the features for one transcript. Num rows = num transcripts, num cols = num features.
        Return: nothing.
        '''
        # Assume all transcripts in the set have the same features in the same order
        if self.transcript_set.get_length() > 0 and self.transcript_set[0].feature_set:
            headers = ['FileID'] + [feat.name for feat in self.transcript_set[0].feature_set]
            with open(output_csv, 'w') as csvfout:
                csvwriter = csv.writer(csvfout, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                csvwriter.writerow(headers)
                for t in self.transcript_set:
                    csvwriter.writerow([functions.get_fileid(t.filename)] + [feat.value for feat in t.feature_set])
