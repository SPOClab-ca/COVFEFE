import bz2
from gensim.corpora import indexedcorpus, mmcorpus, Dictionary
from gensim.models import ldamodel
from scipy import stats
import csv
import numpy as np
import subprocess
import re
import os
import logging

from utils.logger import get_logger

def get_rstHistogram(path, f_trans, path_to_rst_python=None, path_to_rst=None, output_rst_dir=None, NaNvalue=None):
    '''
    path    : string, the path to process
    f_trans : reqd, the actual file to process
    path_to_rst_python : reqd, string. Full path to virtualenv python, for RST
    path_to_rst : reqd, string. Full path to folder with RST's 'parse.py'
    output_rst_dir: reqd, string. Full path to RST's output
    NaNvalue    : opt, the value of NaN used by the feature extractor

    Return: vector of raw counts of RST relations in the following order:
    
            'rst_num_attribution', 'rst_num_background', 'rst_num_cause', 'rst_num_comparison', 'rst_num_condition',
            'rst_num_contrast', 'rst_num_elaboration', 'rst_num_enablement', 'rst_num_evaluation',
            'rst_num_explanation', 'rst_num_joint', 'rst_num_manner-means', 'rst_num_sameUnit',
            'rst_num_summary', 'rst_num_temporal', 'rst_topic-comment

    '''
    # from /p/spoclab/tools/Discourse/RST/src/utils/RST_Classes.py
    relns = ['Attribution','Background','Cause','Comparison','Condition','Contrast','Elaboration','Enablement','Evaluation',
             'Explanation', 'Joint', 'Manner-Means', 'same-unit','Summary','Temporal-sequence','Topic-Change']
    histogram = [0] * 16

    if path_to_rst_python is None or path_to_rst is None or output_rst_dir is None:
        get_logger().log(logging.ERROR, 'RST requires a) its virtual environment, b) the location of parse.py, and c) a place to store its results')
        return [NaNvalue] * 16

    #print [path_to_rst_python, 'parse.py', '-g', '-t', output_rst_dir, path]
    #../../rstenv/bin/python parse.py -g -t ~/foo ../texts/test1.txt
    p = subprocess.Popen([path_to_rst_python, 'parse.py', '-g', '-t', output_rst_dir, path], cwd=path_to_rst); 
    p.wait()

    # RST won't write a file for tasks like fluency, so just return NaN in those cases
    output_file_name = output_rst_dir + '/' + f_trans + '.tree'
    if os.path.exists(output_file_name):
        fparse = open(output_file_name, "r")
        for line in fparse:
            reg = re.match(r'\s*\(([^\[]+)\[', line)
            if reg:
                try: 
                    i = relns.index(reg.group(1))
                    histogram[i] += 1
                except:
                    get_logger().log(logging.ERROR, "%s not in RST relations list" % reg.group(1))
    else:
        histogram = [NaNvalue] * 16

    return histogram


def return_file(fname):
    '''
    Helper function for LDA. Files from training LDA model may be compressed
    '''
    if 'bz2' in fname:
        return bz2.BZ2File(fname)
    else:
        return fname

def get_lda_topics(transcript_utterances, trained_lda_model_filepath, trained_lda_wordids_filepath):
    '''
    Parameters
    transcript_utterances: list of lists of strings (words), each row is a plaintext utterance in the transcript.
    trained_lda_model_filepath: string, path to trained LDA model ('/p/spoclab/models/LDA/lda_model_wiki').
    trained_lda_wordids_filepath: string, path to word IDs of trained LDA model (''/p/spoclab/models/LDA/lda_wordids.txt.bz2).

    Returns:
    topic_probabilities: list of floats, probability of each k topic.
    kurtosis: float, kurtosis of all topic probablities.
    skewness: float, skewness of all topic probabilities.
    entrpy: float, entropy of all topic probabilities.
    '''

    # Get files
    trained_lda_model = return_file(trained_lda_model_filepath)
    trained_lda_wordids = return_file(trained_lda_wordids_filepath)

    # Load LDA model
    lda_model = ldamodel.LdaModel.load(trained_lda_model)

    # Load wordids as a dictionary
    id2word = Dictionary.load_from_text(trained_lda_wordids)

    # Convert transcript of tokens into a BoW document
    document_bow = []
    for transcript_utterance in transcript_utterances:
        document_bow += id2word.doc2bow(transcript_utterance)

    # Get document topics
    doc_topics = lda_model.get_document_topics(document_bow, minimum_probability=0)
    topic_probabilities = [doc_topic[1] for doc_topic in doc_topics]

    skewness = stats.skew(topic_probabilities)
    kurtosis = stats.kurtosis(topic_probabilities)

    # Entropy: SUM(-plog2p)
    entropy = np.sum([-(p * np.log2(p)) for p in topic_probabilities])

    return topic_probabilities, kurtosis, skewness, entropy

