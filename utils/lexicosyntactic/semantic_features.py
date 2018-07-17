""" This module extracts semantic features. """

from nltk.corpus import wordnet as wn
import numpy as np
import logging

from utils.logger import get_logger

def get_wordnet_features(transcript_utterances, brown_information_content=[],
                         semcor_information_content=[], nan_value=-1):

    ''' This function extracts Wordnet features

    Parameters:
    transcript_utterances: list of lists of strings (words), each row is a plaintext utterance in the transcript.
    brown_information_content: wordnet_ic, information content from Brown corpus.
    semcor_information_content: wordnet_ic, information content from SemCor.
    nan_value: int, value for nan (default=-1).

    Returns:
    wordnet_keys: list of strings, names of extracted features.
    wordnet_features: dictionary, mapping feature name to feature value.
    '''

    lSenses = []
    resSimsB = []
    jcnSimsB = []
    linSimsB = []
    resSimsS = []
    jcnSimsS = []
    linSimsS = []
    lcSims = []
    wpSims = []

    for utt in transcript_utterances:
        for word in utt:
            senses = wn.synsets(word)
            if senses:
                lSenses += [senses[0]]

    for iword1 in range(0, len(lSenses)-1):
        for iword2 in range(iword1+1, len(lSenses)):
            try:
                if lSenses[iword1].pos() == lSenses[iword2].pos():
                    resSimsB += [lSenses[iword1].res_similarity(lSenses[iword2], brown_information_content)]
                    jcnSimsB += [lSenses[iword1].jcn_similarity(lSenses[iword2], brown_information_content)]
                    linSimsB += [lSenses[iword1].lin_similarity(lSenses[iword2], brown_information_content)]
                else:
                    resSimsB += []
                    jcnSimsB += []
                    linSimsB += []
            except:
                get_logger().log(logging.ERROR, lSenses[iword1], ' or ', lSenses[iword2], ' does not exist in Brown')
            try:
                if lSenses[iword1].pos() == lSenses[iword2].pos():
                    resSimsS += [lSenses[iword1].res_similarity(lSenses[iword2], semcor_information_content)]
                    jcnSimsS += [lSenses[iword1].jcn_similarity(lSenses[iword2], semcor_information_content)]
                    linSimsS += [lSenses[iword1].lin_similarity(lSenses[iword2], semcor_information_content)]
                else:
                    resSimsS += []
                    jcnSimsS += []
                    linSimsS += []
            except:
                get_logger().log(logging.ERROR, lSenses[iword1], ' or ', lSenses[iword2], ' does not exist in Semcor')

            if lSenses[iword1].pos() == lSenses[iword2].pos():
                lcSims += [lSenses[iword1].lch_similarity(lSenses[iword2], simulate_root=True)]
                wpSims += [lSenses[iword1].wup_similarity(lSenses[iword2], simulate_root=True)]
            else:
                lcSims += []
                wpSims += []

    lcSims = [x for x in lcSims if x is not None]
    wpSims = [x for x in wpSims if x is not None]
    wordnet_keys = ['avg_wn_sim_Res_brown', 'std_wn_sim_Res_brown', 'avg_wn_sim_Res_semcor', 'std_wn_sim_Res_semcor',
                    'avg_wn_sim_JCN_brown', 'std_wn_sim_JCN_brown', 'avg_wn_sim_JCN_semcor', 'std_wn_sim_JCN_semcor',
                    'avg_wn_sim_Lin_brown', 'std_wn_sim_Lin_brown', 'avg_wn_sim_Lin_semcor', 'std_wn_sim_Lin_semcor',
                    'avg_wn_sim_LC', 'std_wn_sim_LC', 'avg_wn_sim_WP', 'std_wn_sim_WP']

    wordnet_features = {}

    wordnet_features['avg_wn_sim_Res_brown'] = np.mean(resSimsB) if resSimsB else nan_value
    wordnet_features['std_wn_sim_Res_brown'] = np.std(resSimsB) if resSimsB else nan_value
    wordnet_features['avg_wn_sim_Res_semcor'] = np.mean(resSimsS) if resSimsS else nan_value
    wordnet_features['std_wn_sim_Res_semcor'] = np.std(resSimsS) if resSimsS else nan_value
    wordnet_features['avg_wn_sim_JCN_brown'] = np.mean(jcnSimsB) if jcnSimsB else nan_value
    wordnet_features['std_wn_sim_JCN_brown'] = np.std(jcnSimsB) if jcnSimsB else nan_value
    wordnet_features['avg_wn_sim_JCN_semcor'] = np.mean(jcnSimsS) if jcnSimsS else nan_value
    wordnet_features['std_wn_sim_JCN_semcor'] = np.std(jcnSimsS) if jcnSimsS else nan_value
    wordnet_features['avg_wn_sim_Lin_brown'] = np.mean(linSimsB) if linSimsB else nan_value
    wordnet_features['std_wn_sim_Lin_brown'] = np.std(linSimsB) if linSimsB else nan_value
    wordnet_features['avg_wn_sim_Lin_semcor'] = np.mean(linSimsS) if linSimsS else nan_value
    wordnet_features['std_wn_sim_Lin_semcor'] = np.std(linSimsS) if linSimsS else nan_value

    wordnet_features['avg_wn_sim_LC'] = np.mean(lcSims) if lcSims else nan_value
    wordnet_features['std_wn_sim_LC'] = np.std(lcSims) if lcSims else nan_value
    wordnet_features['avg_wn_sim_WP'] = np.mean(wpSims) if wpSims else nan_value
    wordnet_features['std_wn_sim_WP'] = np.std(wpSims) if wpSims else nan_value

    return wordnet_keys, wordnet_features
