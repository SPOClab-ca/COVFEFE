""" This module extracts lexical features. """

import collections
import logging
import re
import sys
import math
import subprocess
import numpy as np
import scipy.stats
import scipy.spatial.distance
import nltk.probability
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize
import requests
import json


from utils.logger import get_logger
from utils.lexicosyntactic import functions

def get_wordnet_features(pos_utterances, nan_value):

    ''' This function extracts WordNet features.

    Parameters:
    pos_utterances: list of tuples of strings, (token, POS tag).
    nan_value: int, value for nan

    Returns:
    wordnet_keys: list of strings, names of extracted features.
    wordnet_features: dictionary, mapping feature name to feature value.
    '''

    wordnet_keys = ['avg_max_wn_depth_nn', 'sd_max_wn_depth_nn', 'avg_min_wn_depth_nn', 'sd_min_wn_depth_nn',
                    'avg_wn_ambig_nn', 'sd_wn_ambig_nn', 'kurt_wn_ambig_nn', 'skew_wn_ambig_nn',
                    'avg_max_wn_depth_vb', 'sd_max_wn_depth_vb', 'avg_min_wn_depth_vb', 'sd_min_wn_depth_vb',
                    'avg_wn_ambig_vb', 'sd_wn_ambig_vb', 'kurt_wn_ambig_vb', 'skew_wn_ambig_vb',
                    'avg_max_wn_depth', 'sd_max_wn_depth', 'avg_min_wn_depth', 'sd_min_wn_depth',
                    'avg_wn_ambig', 'sd_wn_ambig', 'kurt_wn_ambig', 'skew_wn_ambig']
    wordnet_features = {}

    wn_ambigs = []
    wn_ambigs_nn = []
    wn_ambigs_vb = []
    wn_max_depths = []
    wn_max_depths_nn = []
    wn_max_depths_vb = []
    wn_min_depths = []
    wn_min_depths_nn = []
    wn_min_depths_vb = []

    for utt in pos_utterances:
        for token_tuple in utt:
            if re.match(r"^NN.*$", token_tuple[1]):
                syns = wn.synsets(token_tuple[0], wn.NOUN)
                if syns:
                    wn_ambigs_nn += [len(syns)]
                    tmp_max = 0.0
                    tmp_min = 0.0
                    for syn in syns:
                        tmp_max += syn.max_depth()
                        tmp_min += syn.min_depth()
                    wn_max_depths_nn += [tmp_max/len(syns)]
                    wn_min_depths_nn += [tmp_min/len(syns)]
            elif re.match(r"^VB.*$", token_tuple[1]):
                syns = wn.synsets(token_tuple[0], wn.VERB)
                if syns:
                    wn_ambigs_vb += [len(syns)]
                    tmp_max = 0.0
                    tmp_min = 0.0
                    for syn in syns:
                        tmp_max += syn.max_depth()
                        tmp_min += syn.min_depth()
                    wn_max_depths_vb += [tmp_max/len(syns)]
                    wn_min_depths_vb += [tmp_min/len(syns)]

            syns = wn.synsets(token_tuple[0])  # this counts ambiguous POS, ignoring the tagger
            if syns:
                wn_ambigs += [len(syns)]
                tmp_max = 0.0
                tmp_min = 0.0
                for syn in syns:
                    tmp_max += syn.max_depth()
                    tmp_min += syn.min_depth()
                wn_max_depths += [tmp_max/len(syns)]
                wn_min_depths += [tmp_min/len(syns)]

    wordnet_features['avg_wn_ambig'] = np.mean(wn_ambigs) if wn_ambigs else nan_value
    wordnet_features['sd_wn_ambig'] = np.std(wn_ambigs) if wn_ambigs else nan_value
    wordnet_features['kurt_wn_ambig'] = scipy.stats.kurtosis(wn_ambigs) if wn_ambigs else nan_value
    wordnet_features['skew_wn_ambig'] = scipy.stats.skew(wn_ambigs) if wn_ambigs else nan_value

    wordnet_features['avg_wn_ambig_nn'] = np.mean(wn_ambigs_nn) if wn_ambigs_nn else nan_value
    wordnet_features['sd_wn_ambig_nn'] = np.std(wn_ambigs_nn) if wn_ambigs_nn else nan_value
    wordnet_features['kurt_wn_ambig_nn'] = scipy.stats.kurtosis(wn_ambigs_nn) if wn_ambigs_nn else nan_value
    wordnet_features['skew_wn_ambig_nn'] = scipy.stats.skew(wn_ambigs_nn) if wn_ambigs_nn else nan_value

    wordnet_features['avg_wn_ambig_vb'] = np.mean(wn_ambigs_vb) if wn_ambigs_vb else nan_value
    wordnet_features['sd_wn_ambig_vb'] = np.std(wn_ambigs_vb) if wn_ambigs_vb else nan_value
    wordnet_features['kurt_wn_ambig_vb'] = scipy.stats.kurtosis(wn_ambigs_vb) if wn_ambigs_vb else nan_value
    wordnet_features['skew_wn_ambig_vb'] = scipy.stats.skew(wn_ambigs_vb) if wn_ambigs_vb else nan_value

    wordnet_features['avg_max_wn_depth'] = np.mean(wn_max_depths) if wn_max_depths else nan_value
    wordnet_features['sd_max_wn_depth'] = np.std(wn_max_depths) if wn_max_depths else nan_value
    wordnet_features['avg_min_wn_depth'] = np.mean(wn_min_depths) if wn_min_depths else nan_value
    wordnet_features['sd_min_wn_depth'] = np.std(wn_min_depths) if wn_min_depths else nan_value

    wordnet_features['avg_max_wn_depth_nn'] = np.mean(wn_max_depths_nn) if wn_max_depths_nn else nan_value
    wordnet_features['sd_max_wn_depth_nn'] = np.std(wn_max_depths_nn) if wn_max_depths_nn else nan_value
    wordnet_features['avg_min_wn_depth_nn'] = np.mean(wn_min_depths_nn) if wn_min_depths_nn else nan_value
    wordnet_features['sd_min_wn_depth_nn'] = np.std(wn_min_depths_nn) if wn_min_depths_nn else nan_value

    wordnet_features['avg_max_wn_depth_vb'] = np.mean(wn_max_depths_vb) if wn_max_depths_vb else nan_value
    wordnet_features['sd_max_wn_depth_vb'] = np.std(wn_max_depths_vb) if wn_max_depths_vb else nan_value
    wordnet_features['avg_min_wn_depth_vb'] = np.mean(wn_min_depths_vb) if wn_min_depths_vb else nan_value
    wordnet_features['sd_min_wn_depth_vb'] = np.std(wn_min_depths_vb) if wn_min_depths_vb else nan_value

    return wordnet_keys, wordnet_features

def get_cosine_distance(transcript_utterances, stopwords, inf_value):

    ''' This function extracts cosine distance features.

    Parameters:
    transcript_utterances: list of lists of strings (words), each row is a plaintext utterance in the transcript.
    stopwords: list of string, words to be removed.
    inf_value: int, value for infinity.

    Returns:
    cosine_keys: list of strings, names of extracted features.
    cosine_features_dict: dictionary, mapping feature name to feature value.
    '''

    cosine_keys = ["ave_cos_dist", "min_cos_dist", "cos_cutoff_00", "cos_cutoff_03", "cos_cutoff_05"]
    cosine_features_dict = {}

    # REPETITION
    # Build a vocab for the transcript
    fdist_vocab = nltk.probability.FreqDist([word for utt in transcript_utterances for word in utt])
    vocab_words = list(fdist_vocab.keys())
    for s in stopwords:
        if s in vocab_words:
            vocab_words.remove(s)

    num_utterances = len(transcript_utterances)

    # Create a word vector for each utterance, N x V
    # where N is the num of utterances and V is the vocab size
    # The vector is 1 if the vocab word is present in the utterance,
    # 0 otherwise (i.e., one hot encoded).
    word_vectors = []
    for i, utt in enumerate(transcript_utterances):
        word_vectors.append(len(vocab_words)*[0]) # init
        for j in range(len(vocab_words)):
            if vocab_words[j] in utt:
                word_vectors[i][j] += 1

    # Calculate cosine DISTANCE between each pair of utterances in
    # this transcript (many entries with small distances means the
    # subject is repeating a lot of words).
    average_dist = 0.0
    min_dist = 1.0
    num_similar_00 = 0.0
    num_similar_03 = 0.0
    num_similar_05 = 0.0
    num_pairs = 0
    for i in range(num_utterances):
        for j in range(i):
            # The norms of the vectors might be zero if the utterance contained only
            # stopwords which were removed above. Only compute cosine distance if the
            # norms are non-zero; ignore the rest.
            norm_i, norm_j = np.linalg.norm(word_vectors[i]), np.linalg.norm(word_vectors[j])
            if norm_i > 0 and norm_j > 0:
                cosine_dist = scipy.spatial.distance.cosine(word_vectors[i], word_vectors[j])
                if math.isnan(cosine_dist):
                    continue
                average_dist += cosine_dist
                num_pairs += 1
                if cosine_dist < min_dist:
                    min_dist = cosine_dist

                # Try different cutoffs for similarity
                if cosine_dist < 0.001: #similarity threshold
                    num_similar_00 += 1
                if cosine_dist <= 0.3: #similarity threshold
                    num_similar_03 += 1
                if cosine_dist <= 0.5: #similarity threshold
                    num_similar_05 += 1

    # The total number of unique utterance pairwise comparisons is <= N*(N-1)/2
    # (could be less if some utterances contain only stopwords and end up empty after
    # stopword removal).
    denom = num_pairs

    if denom >= 1:
        cosine_features = [average_dist * 1.0 / denom,
                           min_dist,
                           num_similar_00 * 1.0 / denom,
                           num_similar_03 * 1.0 / denom,
                           num_similar_05 * 1.0 / denom]
    else:
        # There are either no utterances or a single utterance -- no repetition occurs
        cosine_features = [inf_value, inf_value, 0, 0, 0]

    for ind_feat, feat_name in enumerate(cosine_keys):
        cosine_features_dict[feat_name] = cosine_features[ind_feat]

    return cosine_keys, cosine_features_dict

def get_filler_counts(transcript_utterances_fillers):

    ''' This function extracts filler counts.

    Parameters:
    transcript_utterances_fillers: list of list of strings, transcript utterances with fillers included.

    Returns:
    filler_keys: list of strings, names of extracted features.
    filler_features: dictionary, mapping feature name to feature value.
    '''

    filler_keys = []
    filler_counts = {}

    regex_fillers = {'fillers': re.compile(r'^(?:(?:ah)|(?:eh)|(?:er)|(?:ew)|(?:hm)|(?:mm)|(?:uh)|(?:uhm)|(?:um))$'),
                     'um': re.compile(r'^(?:(?:uhm)|(?:um))$'),
                     'uh': re.compile(r'^(?:(?:ah)|(?:uh))$')}
    filler_keys = regex_fillers.keys()
    filler_counts = collections.defaultdict(int)

    if transcript_utterances_fillers is not None:
        for utt in transcript_utterances_fillers:
            for word in utt:
                for filler_type in filler_keys:
                    if regex_fillers[filler_type].findall(word):
                        filler_counts[filler_type] += 1

    return filler_keys, filler_counts

def get_vocab_richness_measures(pos_tokens, lemmatized_tokens, total_words, inf_value):
    ''' This function extracts vocabulary richness measures:
    Honore statistic, Brunet index, type-token ratio (TTR), moving average type-token ratio (MATTR)

    Parameters:
    pos_tokens: list of tuples of strings, (token, POS_tag) of non-punctuation tokens.
    lemmatized_tokens: list of strings, lemmatized non-punctuation tokens.
    total_words: int, total number of words.
    inf_value: int, infinity value.

    Returns:
    vocab_keys: list of strings, names of extracted features.
    vocab_features: dictionary, mapping feature name to feature value.
    '''

    vocab_keys = ['TTR', 'brunet', 'honore']
    vocab_features = {}

    # MATTR - shift a window over the transcript and compute
    # moving average TTR over each window, then average over all windows
    for window_size in [10, 20, 30, 40, 50]:
        start = 0
        end = window_size
        MATTR = 0

        vocab_features['MATTR_%d' % (window_size)] = 0
        vocab_keys += ['MATTR_%d' % (window_size)]
        while end < len(lemmatized_tokens):
            lem_types = len(set(lemmatized_tokens[start:end]))
            MATTR += 1.0 * lem_types / window_size
            start += 1 # shift window one word at a time
            end += 1
        if start > 0:
            vocab_features['MATTR_%d' % (window_size)] = 1.0 * MATTR / start

    word_types = len(set(pos_tokens)) # same word with different POS = different tokens (confirm with Katie)
    fd_tokens = nltk.probability.FreqDist(pos_tokens)

    # Count number of tokens that occur only once in transcript
    once_words = 0
    for num in fd_tokens.values():
        if num == 1:
            once_words += 1

    try:
        vocab_features["TTR"] = 1.0 * word_types / total_words
        vocab_features["brunet"] = 1.0 * total_words**(word_types**(-0.165)) # Brunet's index - Vlado
    except:
        vocab_features["TTR"] = 0
        vocab_features["brunet"] = 0
    try:
        vocab_features["honore"] = 100.0 * math.log(total_words)/(1.0-1.0*once_words/word_types) # Honore's statistic-Vlado
    except:
        vocab_features["honore"] = inf_value #or infinity ...? (If all words are used only once)

    return vocab_keys, vocab_features

def get_mpqa_norm_features(lemmatized_tokens, mpqa_words, mpqa_types, mpqa_polarities, nan_value):
    ''' This function extracts objectivity polarity measures based on the MPQA lexicon norms:
    strong positive, strong negative, weak positive, weak negative.

    Parameters:
    lemmatized_tokens: list of strings, lemmatized non-punctuation tokens.
    mpqa_words: list of strings, list of words found in the MPQA lexicon.
    mpqa_types: list of strings, type (strong vs negative) of words in the MPQA lexicon.
    mpqa_polarities: list of strings, polarity (positive vs negative) of words in the MPQA lexicon.
    nan_value: int, value of nan.

    Returns:
    mpqa_keys: list of strings, names of extracted features.
    mpqa_features: dictionary, mapping feature name to feature value.
    '''

    mpqa_keys = ['mpqa_strong_positive', 'mpqa_strong_negative', 'mpqa_weak_positive', 'mpqa_weak_negative',
                 'mpqa_num']
    mpqa_features = collections.defaultdict(int)

    for lemmatized_token in lemmatized_tokens:
        if lemmatized_token in mpqa_words:
            mpqa_features["mpqa_num"] += 1
            mpqa_idx = mpqa_words.index(lemmatized_token)
            mpqa_type = mpqa_types[mpqa_idx]
            mpqa_polarity = mpqa_polarities[mpqa_idx]

            if mpqa_type == 'strong' and mpqa_polarity == 'positive':
                mpqa_features["mpqa_strong_positive"] += 1
            elif mpqa_type == 'strong' and mpqa_polarity == 'negative':
                mpqa_features["mpqa_strong_negative"] += 1
            elif mpqa_type == 'weak' and mpqa_polarity == 'positive':
                mpqa_features["mpqa_weak_positive"] += 1
            elif mpqa_type == 'weak' and mpqa_polarity == 'negative':
                mpqa_features["mpqa_weak_negative"] += 1

    # Normalize MPQA subjectivity norms
    if mpqa_features["mpqa_num"] > 0:
        mpqa_features["mpqa_strong_positive"] /= 1.0 * mpqa_features["mpqa_num"]
        mpqa_features["mpqa_strong_negative"] /= 1.0 * mpqa_features["mpqa_num"]
        mpqa_features["mpqa_weak_positive"] /= 1.0 * mpqa_features["mpqa_num"]
        mpqa_features["mpqa_weak_negative"] /= 1.0 * mpqa_features["mpqa_num"]
    else:
        mpqa_features["mpqa_strong_positive"] = nan_value
        mpqa_features["mpqa_strong_negative"] = nan_value
        mpqa_features["mpqa_weak_positive"] = nan_value
        mpqa_features["mpqa_weak_negative"] = nan_value

    return mpqa_keys, mpqa_features

def get_readability_measures(transcript_utterances, prondict, total_words, nan_value):
    ''' This functions extracts readability measures based on the number of syllables:
    Flesch, Flesch-Kincaid, number of syllables per word (average, standard deviation,
    kurtosis, skewness)
    https://datawarrior.wordpress.com/2016/03/29/flesch-kincaid-readability-measure/

    Parameters:
    transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
    prondict: dictionary of valid words for the language, CMU dictionary is used.
    total_words: int, total number of words.
    nan_value: int, value for NaN

    Returns:
    readability_keys: list of strings, names of extracted features.
    readability_features: dictionary, mapping feature name to feature value.
    '''

    readability_keys = ['flesch', 'flesch_kinkaid', 'avg_syll_per_word', 'std_syll_per_word', 'kurt_syll_per_word', 'skew_syll_per_word']
    readability_features = {}

    total_sents = 0
    sylls = []
    for utt in transcript_utterances:
        sutt = ' '.join(utt)
        total_sents += len(sent_tokenize(sutt))
        for w in utt:
            sylls += functions.numsyllables(w, prondict)
    numUnknownProns = len(list(filter(lambda a: a == 0, sylls)))

    try:
        readability_features["flesch"] = 206.835 - 1.015*(total_words-numUnknownProns)/total_sents - 84.6*sum(sylls)/(total_words-numUnknownProns) # TODO
        readability_features["flesch_kinkaid"] = 0.39 * (total_words-numUnknownProns) / total_sents + 11.8 * sum(sylls) / (total_words-numUnknownProns) - 15.59
    except Exception as e:
        readability_features["flesch"] = 0
        readability_features["flesch_kinkaid"] = 0

    readability_features['avg_syll_per_word'] = np.mean(sylls) if sylls else nan_value
    readability_features['std_syll_per_word'] = np.std(sylls) if sylls else nan_value
    readability_features['kurt_syll_per_word'] = scipy.stats.kurtosis(sylls) if sylls else nan_value
    readability_features['skew_syll_per_word'] = scipy.stats.skew(sylls) if sylls else nan_value

    return readability_keys, readability_features

def get_liwc_features(transcript_utterances, nan_value):
    ''' This functions extracts sentiment analysis features using the Stanford sentiment analysis tool.                                                                                                  
    Parameters:                                                                                                                                                                                            
    transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.                                                                                        
    nan_value: int, value for NaN    
    '''
    liwc_keys = [];
    liwc_features = {};

    sServer = 'https://app.receptiviti.com'
    api_key = '***REMOVED***'
    api_secret = '***REMOVED***'
    iPerson = '***REMOVED***'
    url = '%s/v2/api/person/%s/contents' % (sServer, iPerson)

    allUtts = ''
    for utt in transcript_utterances:
        sutt = ' '.join(utt)
        allUtts += sutt


    headers = {
        "X-API-KEY": api_key,
        "X-API-SECRET-KEY": api_secret,
        'Content-Type': 'application/json'
    }

    payload = {
        'content_handle' : '0',
        'language_content' : allUtts,
        'content_source' : 0
    }

    response = requests.post(url, headers=headers, data=payload)

    if response.status_code != 200:
        get_logger().log(logging.ERROR, "Response code %s from Receptiviti" % response.status_code)
        liwc_keys = ['liwc_fail', 'receptiviti_fail']
        liwc_features['liwc_fail'] = nan_value
        liwc_features['receptiviti_fail'] = nan_value
    else:
        for key in response.json()['receptiviti_scores']['raw_scores']:
            skey = 'receptiviti_%s' % key
            liwc_keys += [skey]
            liwc_features[skey] = response.json()['receptiviti_scores']['raw_scores'][key]
        for key in response.json()['liwc_scores']['categories']:
            skey = 'liwc_%s' % key
            liwc_keys += [skey]
            liwc_features[skey] = response.json()['liwc_scores']['categories'][key]


    return liwc_keys, liwc_features 

def get_stanford_sentiment_features(transcript_utterances, path_to_stanford_cp, nan_value):
    ''' This functions extracts sentiment analysis features using the Stanford sentiment analysis tool.

    Parameters:
    transcript_utterances : list of lists of strings (words); each row is a plaintext utterance in the transcript.
    path_to_stanford_cp: string, path to Stanford corenlp
    nan_value: int, value for NaN

    Returns:
    stanford_keys: list of strings, names of extracted features.
    stanford_features: dictionary, mapping feature name to feature value.
    '''
    stanford_keys = ['mean_stanford_sentiment_veryneg', 'mean_stanford_sentiment_neg',
                     'mean_stanford_sentiment_neutral', 'mean_stanford_sentiment_pos', 'mean_stanford_sentiment_verypos',
                     'std_stanford_sentiment_veryneg', 'std_stanford_sentiment_neg',
                     'std_stanford_sentiment_neutral', 'std_stanford_sentiment_pos', 'std_stanford_sentiment_verypos']
    stanford_features = {}

    sentiments = []
    try:
        for utt in transcript_utterances:
            sentence = ' '.join(utt)
            cmd = 'java -cp "%s" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline -output PROBABILITIES -stdin' % path_to_stanford_cp
            #print cmd
            proc = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
            proc.stdin.write(sentence.encode())
            proc.stdin.close()
            result = proc.stdout.read().decode()
            lines = result.splitlines()
            line = re.split(r'\s+', lines[1].rstrip('\t'))
            sentiments = np.vstack((sentiments, np.array(line[2:]))) if len(sentiments) > 0 else np.array(line[2:])
        sentiments = np.reshape(sentiments, (-1, 5))
        sentiments = sentiments.astype(np.float)
        if sentiments.shape[0] > 1:
            msentiments = np.mean(sentiments, axis=0)
            ssentiments = np.std(sentiments, axis=0)
        else:
            msentiments = sentiments.flatten()
            ssentiments = np.array([0]*5)

        stanford_features['mean_stanford_sentiment_veryneg'] = msentiments[0] if msentiments.size >0 else nan_value
        stanford_features['mean_stanford_sentiment_neg'] = msentiments[1] if msentiments.size >1 else nan_value
        stanford_features['mean_stanford_sentiment_neutral'] = msentiments[2] if msentiments.size>2 else nan_value
        stanford_features['mean_stanford_sentiment_pos'] = msentiments[3] if msentiments.size>3 else nan_value
        stanford_features['mean_stanford_sentiment_verypos'] = msentiments[4] if msentiments.size>4 else nan_value
        stanford_features['std_stanford_sentiment_veryneg'] = ssentiments[0] if ssentiments.size>0 else nan_value
        stanford_features['std_stanford_sentiment_neg'] = ssentiments[1] if ssentiments.size>1 else nan_value
        stanford_features['std_stanford_sentiment_neutral'] = ssentiments[2] if ssentiments.size>2 else nan_value
        stanford_features['std_stanford_sentiment_pos'] = ssentiments[3] if ssentiments.size>3 else nan_value
        stanford_features['std_stanford_sentiment_verypos'] = ssentiments[4] if ssentiments.size>4 else nan_value

    except (NameError,) as e:
        msg = 'Cannot run Stanford sentiment analysis, using %s' % path_to_stanford_cp
        msg = msg + "\n" + re.findall("name '(\w+)' is not defined " + str(e))[0]
        msg = msg + "\n" + "Error > " + sys.exc_info()[0]

        get_logger().log(logging.ERROR, msg)

        for stanford_key in stanford_keys:
            stanford_features[stanford_key] = nan_value
    except (IndexError) as e:
        get_logger().log(logging.ERROR, 'Problem with Stanford sentiment analysis output file' + str(e))
        for stanford_key in stanford_keys:
            stanford_features[stanford_key] = nan_value

    return stanford_keys, stanford_features

def get_pos_features(pos_utterances, total_words, lemmatizer,
                     norms_freq, norms_image, norms_anew, norms_warringer,
                     function_tags, inflected_verb_tags, light_verbs, subordinate,
                     demonstratives, dictionary_words, word_exceptions, inf_value, nan_value,
                     get_pos_counts, get_pos_ratios, get_frequency_norms, get_image_norms, get_anew_norms,
                     get_warringer_norms, get_density):

    ''' This general purpose functions extracts POS features.
    Parameters:
    pos_utterances : list of lists of tuples of (token, POStag); each row is an utterance. No filled pauses.
    total_words: int, total number of words.
    lemmatizer: WordNet Lemmatizer.
    norms_freq: lexical frequency norms.
    norms_image: norms for age-of-acquisition, imageability, familiarity.
    norms_anew: norms for Anew (valence, arousal, dominance).
    norms_warringer: norms for Warringer (valence, arousal, dominance).
    function_tags: list of strings, POS tags for function words.
    inflected_verb_tags: list of strings, POS tags for inflected verbs.
    light_verbs: list of strings, light verbs.
    subordinate:
    demonstratives:
    dictionary_words:
    word_exceptions:
    inf_value:
    nan_value:
    get_pos_counts: boolean, return POS counts if True.
    get_pos_ratios: boolean, return POS ratios if True.
    get_frequency_norms: boolean, return frequency values if True.
    get_image_norms: boolean, return image values (age-of-acquisition, imageability, familiarity) if True.
    get_anew_norms: boolean, return Anew values (valence, arousal, dominance) if True.
    get_warringer_norms: boolean, return Warringer values (valence, arousal, dominance) if True.
    get_density: boolean, return density values if True.

    Returns:
    '''

    pos_keys = ['word_length', 'NID']
    if get_pos_counts:
        pos_keys += ['nouns', 'verbs', 'inflected_verbs', 'light', 'function', 'pronouns', 'determiners', 'adverbs', 'adjectives', 'prepositions',
                     'coordinate', 'subordinate', 'demonstratives']
    if get_pos_ratios:
        pos_keys += ['nvratio', 'prp_ratio', 'noun_ratio', 'sub_coord_ratio']
    if get_frequency_norms:
        pos_keys += ['frequency', 'noun_frequency', 'verb_frequency']
    if get_image_norms:
        pos_keys += ['aoa', 'imageability', 'familiarity',
                     'noun_aoa', 'noun_imageability', 'noun_familiarity',
                     'verb_aoa', 'verb_imageability', 'verb_familiarity']
    if get_anew_norms:
        pos_keys += ['noun_anew_val_mean', 'noun_anew_val_std', 'noun_anew_aro_mean', 'noun_anew_aro_std', 'noun_anew_dom_mean', 'noun_anew_dom_std',
                     'verb_anew_val_mean', 'verb_anew_val_std', 'verb_anew_aro_mean', 'verb_anew_aro_std', 'verb_anew_dom_mean', 'verb_anew_dom_std',
                     'anew_val_mean', 'anew_val_std', 'anew_aro_mean', 'anew_aro_std', 'anew_dom_mean', 'anew_dom_std']
    if get_warringer_norms:
        pos_keys += ['warr_val_mean', 'warr_val_std', 'warr_val_rat', 'warr_aro_mean', 'warr_aro_std',
                     'warr_aro_rat', 'warr_dom_mean', 'warr_dom_std', 'warr_dom_rat',
                     'warr_val_mean_nn', 'warr_val_std_nn', 'warr_val_rat_nn', 'warr_aro_mean_nn', 'warr_aro_std_nn',
                     'warr_aro_rat_nn', 'warr_dom_mean_nn', 'warr_dom_std_nn', 'warr_dom_rat_nn',
                     'warr_val_mean_vb', 'warr_val_std_vb', 'warr_val_rat_vb', 'warr_aro_mean_vb', 'warr_aro_std_vb',
                     'warr_aro_rat_vb', 'warr_dom_mean_vb', 'warr_dom_std_vb', 'warr_dom_rat_vb']
    if get_density:
        pos_keys += ['prop_density', 'content_density']
    pos_features = collections.defaultdict(int)

    # Filter out any punctuation tags
    regex_pos_content = re.compile(r'^[a-zA-Z$]+$')
    pos_tokens = [] # list of tuples of (token, POStag) of non-punctuation tokens
    lemmatized_tokens = [] # list of lemmatized non-punctuation tokens

    if get_density:
        pos_features['prop_density'] = 0
        pos_features['content_density'] = 0

    for utt in pos_utterances:
        for token_tuple in utt:
            # If the POS tag is not that of punctuation, add to tokens
            if regex_pos_content.findall(token_tuple[1]):
                pos_tokens += [token_tuple]

                pos_features['word_length'] += len(token_tuple[0])

                if token_tuple[0] not in dictionary_words and token_tuple[0] not in word_exceptions:
                    pos_features['NID'] += 1

                # Lemmatize according to the type of the word
                lemmatized_token = lemmatizer.lemmatize(token_tuple[0], functions.pos_treebank2wordnet(token_tuple[1]))
                lemmatized_tokens += [lemmatized_token]

                if get_density:
                    if re.match(r"^(NN|VB|JJ|RB|SYM).*$", token_tuple[1]):
                        pos_features['content_density'] += 1
                    if re.match(r"^(VB|JJ|RB|IN|CC).*$", token_tuple[1]):
                        pos_features['prop_density'] += 1

                # Count POS tags
                if re.match(r"^NN.*$", token_tuple[1]):
                    pos_features['nouns'] += 1

                    if get_frequency_norms and token_tuple[0] in norms_freq:
                        pos_features["noun_frequency"] += float(norms_freq[token_tuple[0]][5]) # use log10WF
                        pos_features["noun_freq_num"] += 1

                    if get_image_norms and lemmatized_token in norms_image:
                        pos_features["noun_aoa"] += float(norms_image[lemmatized_token][0])
                        pos_features["noun_imageability"] += float(norms_image[lemmatized_token][1])
                        pos_features["noun_familiarity"] += float(norms_image[lemmatized_token][2])
                        pos_features["noun_img_num"] += 1

                    if get_anew_norms and lemmatized_token in norms_anew:
                        pos_features["noun_anew_val_mean"] += float(norms_anew[lemmatized_token][0])
                        pos_features["noun_anew_val_std"] += float(norms_anew[lemmatized_token][1])
                        pos_features["noun_anew_aro_mean"] += float(norms_anew[lemmatized_token][2])
                        pos_features["noun_anew_aro_std"] += float(norms_anew[lemmatized_token][3])
                        pos_features["noun_anew_dom_mean"] += float(norms_anew[lemmatized_token][4])
                        pos_features["noun_anew_dom_std"] += float(norms_anew[lemmatized_token][5])
                        pos_features["noun_anew_num"] += 1

                    if get_warringer_norms and lemmatized_token in norms_warringer:
                        pos_features["warr_val_mean_nn"] += float(norms_warringer[lemmatized_token][0])
                        pos_features["warr_val_std_nn"] += float(norms_warringer[lemmatized_token][1])
                        pos_features["warr_val_rat_nn"] += float(norms_warringer[lemmatized_token][2])
                        pos_features["warr_aro_mean_nn"] += float(norms_warringer[lemmatized_token][3])
                        pos_features["warr_aro_std_nn"] += float(norms_warringer[lemmatized_token][4])
                        pos_features["warr_aro_rat_nn"] += float(norms_warringer[lemmatized_token][5])
                        pos_features["warr_dom_mean_nn"] += float(norms_warringer[lemmatized_token][6])
                        pos_features["warr_dom_std_nn"] += float(norms_warringer[lemmatized_token][7])
                        pos_features["warr_dom_rat_nn"] += float(norms_warringer[lemmatized_token][8])
                        pos_features["noun_warr_num"] += 1

                elif re.match(r'^V.*$', token_tuple[1]):
                    pos_features['verbs'] += 1

                    if token_tuple[1] in inflected_verb_tags:
                        pos_features['inflected_verbs'] += 1

                    if lemmatized_token in light_verbs:
                        pos_features['light'] += 1

                    if get_frequency_norms and token_tuple[0] in norms_freq:
                        pos_features["verb_frequency"] += float(norms_freq[token_tuple[0]][5]) # use log10WF
                        pos_features["verb_freq_num"] += 1

                    if get_image_norms and lemmatized_token in norms_image:
                        pos_features["verb_aoa"] += float(norms_image[lemmatized_token][0])
                        pos_features["verb_imageability"] += float(norms_image[lemmatized_token][1])
                        pos_features["verb_familiarity"] += float(norms_image[lemmatized_token][2])
                        pos_features["verb_img_num"] += 1

                    if get_anew_norms and lemmatized_token in norms_anew:
                        pos_features["verb_anew_val_mean"] += float(norms_anew[lemmatized_token][0])
                        pos_features["verb_anew_val_std"] += float(norms_anew[lemmatized_token][1])
                        pos_features["verb_anew_aro_mean"] += float(norms_anew[lemmatized_token][2])
                        pos_features["verb_anew_aro_std"] += float(norms_anew[lemmatized_token][3])
                        pos_features["verb_anew_dom_mean"] += float(norms_anew[lemmatized_token][4])
                        pos_features["verb_anew_dom_std"] += float(norms_anew[lemmatized_token][5])
                        pos_features["verb_anew_num"] += 1

                    if get_warringer_norms and lemmatized_token in norms_warringer:
                        pos_features["warr_val_mean_vb"] += float(norms_warringer[lemmatized_token][0])
                        pos_features["warr_val_std_vb"] += float(norms_warringer[lemmatized_token][1])
                        pos_features["warr_val_rat_vb"] += float(norms_warringer[lemmatized_token][2])
                        pos_features["warr_aro_mean_vb"] += float(norms_warringer[lemmatized_token][3])
                        pos_features["warr_aro_std_vb"] += float(norms_warringer[lemmatized_token][4])
                        pos_features["warr_aro_rat_vb"] += float(norms_warringer[lemmatized_token][5])
                        pos_features["warr_dom_mean_vb"] += float(norms_warringer[lemmatized_token][6])
                        pos_features["warr_dom_std_vb"] += float(norms_warringer[lemmatized_token][7])
                        pos_features["warr_dom_rat_vb"] += float(norms_warringer[lemmatized_token][8])
                        pos_features["verb_warr_num"] += 1

                else:
                    if token_tuple[1] in function_tags:
                        pos_features['function'] += 1

                    if re.match(r'^PRP.*$', token_tuple[1]):
                        pos_features['pronouns'] += 1
                    elif re.match(r"^DT$", token_tuple[1]):
                        pos_features["determiners"] += 1
                    elif re.match(r"^RB.*$", token_tuple[1]): #adverb
                        pos_features["adverbs"] += 1
                    elif re.match(r"^JJ.*$", token_tuple[1]): #adjective
                        pos_features["adjectives"] += 1
                    elif re.match(r"^IN$", token_tuple[1]):
                        pos_features["prepositions"] += 1
                    elif re.match(r"^CC$", token_tuple[1]):
                        pos_features["coordinate"] += 1

                if token_tuple[0] in subordinate:
                    if token_tuple[1] in ["IN", "WRB", "WP"]:
                        pos_features["subordinate"] += 1

                if token_tuple[0] in demonstratives:
                    pos_features["demonstratives"] += 1

                if get_frequency_norms and token_tuple[0] in norms_freq: #note: frequencies are not lemmatized
                    pos_features["frequency"] += float(norms_freq[token_tuple[0]][5]) # use log10WF
                    pos_features["freq_num"] += 1

                if get_image_norms and lemmatized_token in norms_image:
                    pos_features["aoa"] += float(norms_image[lemmatized_token][0])
                    pos_features["imageability"] += float(norms_image[lemmatized_token][1])
                    pos_features["familiarity"] += float(norms_image[lemmatized_token][2])
                    pos_features["img_num"] += 1

                if get_anew_norms and lemmatized_token in norms_anew:
                    pos_features["anew_val_mean"] += float(norms_anew[lemmatized_token][0])
                    pos_features["anew_val_std"] += float(norms_anew[lemmatized_token][1])
                    pos_features["anew_aro_mean"] += float(norms_anew[lemmatized_token][2])
                    pos_features["anew_aro_std"] += float(norms_anew[lemmatized_token][3])
                    pos_features["anew_dom_mean"] += float(norms_anew[lemmatized_token][4])
                    pos_features["anew_dom_std"] += float(norms_anew[lemmatized_token][5])
                    pos_features["anew_num"] += 1

                if get_warringer_norms and lemmatized_token in norms_warringer:
                    pos_features["warr_val_mean"] += float(norms_warringer[lemmatized_token][0])
                    pos_features["warr_val_std"] += float(norms_warringer[lemmatized_token][1])
                    pos_features["warr_val_rat"] += float(norms_warringer[lemmatized_token][2])
                    pos_features["warr_aro_mean"] += float(norms_warringer[lemmatized_token][3])
                    pos_features["warr_aro_std"] += float(norms_warringer[lemmatized_token][4])
                    pos_features["warr_aro_rat"] += float(norms_warringer[lemmatized_token][5])
                    pos_features["warr_dom_mean"] += float(norms_warringer[lemmatized_token][6])
                    pos_features["warr_dom_std"] += float(norms_warringer[lemmatized_token][7])
                    pos_features["warr_dom_rat"] += float(norms_warringer[lemmatized_token][8])
                    pos_features["warr_num"] += 1

    # Compute verb ratios, and noun to verb ratio
    if pos_features["verbs"] > 0:
        pos_features["nvratio"] = 1.0 * pos_features["nouns"] / pos_features["verbs"]
        pos_features["light"] = 1.0 * pos_features["light"] / pos_features["verbs"]
        pos_features["inflected_verbs"] = 1.0 * pos_features["inflected_verbs"] / pos_features["verbs"]
    else:
        if pos_features["nouns"] > 0:
            pos_features["nvratio"] = inf_value
        else:
            pos_features["nvratio"] = nan_value
        pos_features["light"] = 0
        pos_features["inflected_verbs"] = 0

    # Compute noun ratios (pronouns to pronoun+nouns, and nouns to noun+verb)
    if pos_features["nouns"] > 0:
        pos_features["prp_ratio"] = 1.0 * pos_features["pronouns"] / (pos_features["pronouns"] + pos_features["nouns"])
        pos_features["noun_ratio"] = 1.0 * pos_features["nouns"] / (pos_features["verbs"] + pos_features["nouns"])
    else:
        if pos_features["pronouns"] > 0:
            pos_features["prp_ratio"] = 1.0 * pos_features["pronouns"] / (pos_features["pronouns"] + pos_features["nouns"])
        else:
            pos_features["prp_ratio"] = nan_value # NaN? 0/0 - no nouns and no pronouns

        if pos_features["verbs"] > 0:
            pos_features["noun_ratio"] = 1.0 * pos_features["nouns"]/(pos_features["verbs"] + pos_features["nouns"])
        else:
            pos_features["noun_ratio"] = nan_value # NaN? 0/0 - no nouns and no verbs

    # Compute conjunction ratios
    if pos_features["coordinate"] > 0:
        pos_features["sub_coord_ratio"] = 1.0 * pos_features["subordinate"] / pos_features["coordinate"]
    else:
        if pos_features['subordinate'] > 0:
            pos_features["sub_coord_ratio"] = inf_value
        else:
            pos_features['sub_coord_ratio'] = nan_value # NaN? 0/0 - no subord and no coord conjunctions

    if pos_features['prop_density'] > 0:
        pos_features['prop_density'] /= total_words
    else:
        pos_features['prop_density'] = nan_value

    if pos_features['content_density'] > 0:
        pos_features['content_density'] /= total_words
    else:
        pos_features['content_density'] = nan_value

    # Normalize all age of acquisition, imageability, familiarity norms
    if pos_features["img_num"] > 0:
        pos_features["aoa"] = 1.0 * pos_features["aoa"] / pos_features["img_num"]
        pos_features["imageability"] = 1.0 * pos_features["imageability"] / pos_features["img_num"]
        pos_features["familiarity"] = 1.0 * pos_features["familiarity"] / pos_features["img_num"]
    else: # no words with imageability norms
        pos_features["aoa"] = nan_value
        pos_features["imageability"] = nan_value
        pos_features["familiarity"] = nan_value

    # Normalize all age of acquisition, imageability, familiarity norms for nouns
    if pos_features["noun_img_num"] > 0:
        pos_features["noun_aoa"] = 1.0 * pos_features["noun_aoa"] / pos_features["noun_img_num"]
        pos_features["noun_imageability"] = 1.0 * pos_features["noun_imageability"] / pos_features["noun_img_num"]
        pos_features["noun_familiarity"] = 1.0 * pos_features["noun_familiarity"] / pos_features["noun_img_num"]
    else:
        pos_features["noun_aoa"] = nan_value
        pos_features["noun_imageability"] = nan_value
        pos_features["noun_familiarity"] = nan_value

    # Normalize all age of acquisition, imageability, familiarity norms for verbs
    if pos_features["verb_img_num"] > 0:
        pos_features["verb_aoa"] = 1.0 * pos_features["verb_aoa"] / pos_features["verb_img_num"]
        pos_features["verb_imageability"] = 1.0 * pos_features["verb_imageability"] / pos_features["verb_img_num"]
        pos_features["verb_familiarity"] = 1.0 * pos_features["verb_familiarity"] / pos_features["verb_img_num"]
    else:
        pos_features["verb_aoa"] = nan_value
        pos_features["verb_imageability"] = nan_value
        pos_features["verb_familiarity"] = nan_value

    # Normalize all anew norms
    if pos_features["anew_num"] > 0:
        pos_features["anew_val_mean"] = 1.0 * pos_features["anew_val_mean"] / pos_features["anew_num"]
        pos_features["anew_val_std"] = 1.0 * pos_features["anew_val_std"] / pos_features["anew_num"]
        pos_features["anew_aro_mean"] = 1.0 * pos_features["anew_aro_mean"] / pos_features["anew_num"]
        pos_features["anew_aro_std"] = 1.0 * pos_features["anew_aro_std"] / pos_features["anew_num"]
        pos_features["anew_dom_mean"] = 1.0 * pos_features["anew_dom_mean"] / pos_features["anew_num"]
        pos_features["anew_dom_std"] = 1.0 * pos_features["anew_dom_std"] / pos_features["anew_num"]
    else: # no words with imageability norms
        pos_features["anew_val_mean"] = nan_value
        pos_features["anew_val_std"] = nan_value
        pos_features["anew_aro_mean"] = nan_value
        pos_features["anew_aro_std"] = nan_value
        pos_features["anew_dom_mean"] = nan_value
        pos_features["anew_dom_std"] = nan_value

    # Normalize all anew norms for nouns
    if pos_features["noun_anew_num"] > 0:
        pos_features["noun_anew_val_mean"] = 1.0 * pos_features["noun_anew_val_mean"] / pos_features["noun_anew_num"]
        pos_features["noun_anew_val_std"] = 1.0 * pos_features["noun_anew_val_std"] / pos_features["noun_anew_num"]
        pos_features["noun_anew_aro_mean"] = 1.0 * pos_features["noun_anew_aro_mean"] / pos_features["noun_anew_num"]
        pos_features["noun_anew_aro_std"] = 1.0 * pos_features["noun_anew_aro_std"] / pos_features["noun_anew_num"]
        pos_features["noun_anew_dom_mean"] = 1.0 * pos_features["noun_anew_dom_mean"] / pos_features["noun_anew_num"]
        pos_features["noun_anew_dom_std"] = 1.0 * pos_features["noun_anew_dom_std"] / pos_features["noun_anew_num"]
    else: # no nouns with anew norms
        pos_features["noun_anew_val_mean"] = nan_value
        pos_features["noun_anew_val_std"] = nan_value
        pos_features["noun_anew_aro_mean"] = nan_value
        pos_features["noun_anew_aro_std"] = nan_value
        pos_features["noun_anew_dom_mean"] = nan_value
        pos_features["noun_anew_dom_std"] = nan_value

    # Normalize all anew norms for verbs
    if pos_features["verb_anew_num"] > 0:
        pos_features["verb_anew_val_mean"] = 1.0 * pos_features["verb_anew_val_mean"] / pos_features["verb_anew_num"]
        pos_features["verb_anew_val_std"] = 1.0 * pos_features["verb_anew_val_std"] / pos_features["verb_anew_num"]
        pos_features["verb_anew_aro_mean"] = 1.0 * pos_features["verb_anew_aro_mean"] / pos_features["verb_anew_num"]
        pos_features["verb_anew_aro_std"] = 1.0 * pos_features["verb_anew_aro_std"] / pos_features["verb_anew_num"]
        pos_features["verb_anew_dom_mean"] = 1.0 * pos_features["verb_anew_dom_mean"] / pos_features["verb_anew_num"]
        pos_features["verb_anew_dom_std"] = 1.0 * pos_features["verb_anew_dom_std"] / pos_features["verb_anew_num"]
    else: # no verbs with anew norms
        pos_features["verb_anew_val_mean"] = nan_value
        pos_features["verb_anew_val_std"] = nan_value
        pos_features["verb_anew_aro_mean"] = nan_value
        pos_features["verb_anew_aro_std"] = nan_value
        pos_features["verb_anew_dom_mean"] = nan_value
        pos_features["verb_anew_dom_std"] = nan_value

    # Normalize all warr norms
    if pos_features["warr_num"] > 0:
        pos_features["warr_val_mean"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_val_std"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_val_rat"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_aro_mean"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_aro_std"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_aro_rat"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_dom_mean"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_dom_std"] /= 1.0*pos_features["warr_num"]
        pos_features["warr_dom_rat"] /= 1.0*pos_features["warr_num"]
    else: # no words with warringer norms
        pos_features["warr_val_mean"] = nan_value
        pos_features["warr_val_std"] = nan_value
        pos_features["warr_val_rat"] = nan_value
        pos_features["warr_aro_mean"] = nan_value
        pos_features["warr_aro_std"] = nan_value
        pos_features["warr_aro_rat"] = nan_value
        pos_features["warr_dom_mean"] = nan_value
        pos_features["warr_dom_std"] = nan_value
        pos_features["warr_dom_rat"] = nan_value

    # Normalize all warr norms for nouns
    if pos_features["noun_warr_num"] > 0:
        pos_features["warr_val_mean_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_val_std_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_val_rat_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_aro_mean_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_aro_std_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_aro_rat_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_dom_mean_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_dom_std_nn"] /= 1.0*pos_features["noun_warr_num"]
        pos_features["warr_dom_rat_nn"] /= 1.0*pos_features["noun_warr_num"]
    else: # no nouns with warringer norms
        pos_features["warr_val_mean_nn"] = nan_value
        pos_features["warr_val_std_nn"] = nan_value
        pos_features["warr_val_rat_nn"] = nan_value
        pos_features["warr_aro_mean_nn"] = nan_value
        pos_features["warr_aro_std_nn"] = nan_value
        pos_features["warr_aro_rat_nn"] = nan_value
        pos_features["warr_dom_mean_nn"] = nan_value
        pos_features["warr_dom_std_nn"] = nan_value
        pos_features["warr_dom_rat_nn"] = nan_value

    # Normalize all warr norms for verbs
    if pos_features["verb_warr_num"] > 0:
        pos_features["warr_val_mean_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_val_std_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_val_rat_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_aro_mean_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_aro_std_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_aro_rat_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_dom_mean_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_dom_std_vb"] /= 1.0*pos_features["verb_warr_num"]
        pos_features["warr_dom_rat_vb"] /= 1.0*pos_features["verb_warr_num"]
    else: # no verbs with warringer norms
        pos_features["warr_val_mean_vb"] = nan_value
        pos_features["warr_val_std_vb"] = nan_value
        pos_features["warr_val_rat_vb"] = nan_value
        pos_features["warr_aro_mean_vb"] = nan_value
        pos_features["warr_aro_std_vb"] = nan_value
        pos_features["warr_aro_rat_vb"] = nan_value
        pos_features["warr_dom_mean_vb"] = nan_value
        pos_features["warr_dom_std_vb"] = nan_value
        pos_features["warr_dom_rat_vb"] = nan_value

    # Normalize frequency norms
    if pos_features["freq_num"] > 0:
        pos_features["frequency"] = 1.0 * pos_features["frequency"] / pos_features["freq_num"]
    else:
        pos_features["frequency"] = nan_value

    # Normalize frequency norms for nouns
    if pos_features["noun_freq_num"] > 0:
        pos_features["noun_frequency"] = 1.0 * pos_features["noun_frequency"] / pos_features["noun_freq_num"]
    else:
        pos_features["noun_frequency"] = nan_value

    # Normalize frequency norms for verbs
    if pos_features["verb_freq_num"] > 0:
        pos_features["verb_frequency"] = 1.0 * pos_features["verb_frequency"] / pos_features["verb_freq_num"]
    else:
        pos_features["verb_frequency"] = nan_value

    return pos_keys, pos_features
