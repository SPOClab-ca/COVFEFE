
import nltk
import os
import re


def get_filename(full_path):
    '''Given a full absolute path to a file, return the filename only (including extension). 
    If path separators are not present, return the original argument.'''
    # Automatically detect the default OS path separator (Windows-style or 
    # Unix-style).
    if full_path.find(os.sep) != -1 and full_path.rfind(os.sep) < len(full_path)-1:
        return full_path[full_path.rfind(os.sep)+1:]
    else:
        return full_path

def get_fileid(filename):
    return filename[:filename.rfind('.')]

def get_subject_sample(file_id):
    '''Given a string, file_id, in the format 'subjectID_sessionID.ext' or 'subjectID-sessionID.ext' 
    or 'subjectID-sessionID-speakerID.ext', return (subjectID, visitID) or (subjectID, visitID, speakerID) 
    as a tuple of strings. If the string is empty or does not follow the expected format, return None.'''
    
    regex_fileformat = re.compile(r'^(?P<subjectID>[0-9a-zA-Z]+)[_-](?P<sessionID>[0-9a-zA-Z]+)[.](?:[0-9a-zA-Z]+)$')
    regex_extendedformat = re.compile(r'^(?P<subjectID>[0-9a-zA-Z]+)[_-](?P<sessionID>[0-9a-zA-Z]+)[-_](?P<speakerID>[0-9a-zA-Z]+)[.](?:[0-9a-zA-Z]+)$')
    if regex_fileformat.findall(file_id):
        return regex_fileformat.findall(file_id)[0]
    elif regex_extendedformat.findall(file_id):
        return regex_extendedformat.findall(file_id)[0]
    return None

def get_frequency_norms(path_to_norms=None):
    """Parameters:
    path_to_norms : optional, string. Full path, including filename, of the frequency norms.
    
    Return dictionary of SUBTL frequencies, keys = words, values = list of norms.""" 
    
    if path_to_norms is not None:
        source_norms = path_to_norms
    else:
        source_norms = os.path.abspath('../feature_extraction/text/frequencies.txt')
    
    with open(source_norms, "r") as fin:
        f = fin.readlines()
        f = f[1:] #skip header
    
        freq = {}
        for line in f:
            l = line.strip().split()
            if len(l) == 0:
                continue
            freq[l[0].lower()] = l[1:] #returns whole line -- usually just use Log10WF
    
        return freq
    return None

def get_warringer_norms(path_to_norms=None):
    """Parameters:
    path_to_norms : optional, string. Full path, including filename, of the Warringer norms.
    
    Return dictionary of warringer norms, order: [warr.V.Mean.Sum,warr.V.SD.Sum,warr.V.Rat.Sum,warr.A.Mean.Sum,warr.A.SD.Sum,warr.A.Rat.Sum,warr.D.Mean.Sum,warr.D.SD.Sum,warr.D.Rat.Sum]""" 

    if path_to_norms is not None:
        source_norms = path_to_norms
    else:
        source_norms = os.path.abspath('TODOme')

    with open(source_norms, "r") as fin:
        f = fin.readlines()
        f = f[1:] # skip header

        warr = {}
        for line in f:
            l = line.strip().split(',')
            warr[l[1]] = l[2:11]

        return warr
    return None

def get_anew_norms(path_to_norms=None):
    """Parameters:
    path_to_norms : optional, string. Full path, including filename, of the ANEW norms.
    
    Return dictionary of ANEW norms, order: [ValMn ValSD AroMn AroSD DomMn DomSD]""" 

    if path_to_norms is not None:
        source_norms = path_to_norms
    else:
        source_norms = os.path.abspath('TODOme')

    with open(source_norms, "r") as fin:
        f = fin.readlines()
        f = f[1:] #skip header

        anew = {}
        for line in f:
            l = line.strip().split()
            anew[l[0]] = l[2:8]

        return anew
    return None


def get_imageability_norms(path_to_norms=None):
    """Parameters:
    path_to_norms : optional, string. Full path, including filename, of the imageability norms.
    
    Return dictionary of Gilooly-Logie-Bristol norms, order: [AoA,imageability,familiarity]""" 
    
    if path_to_norms is not None:
        source_norms = path_to_norms
    else:
        source_norms = os.path.abspath('../feature_extraction/text/image.txt')
        
    with open(source_norms, "r") as fin:
        f = fin.readlines()
        f = f[1:] #skip header (1 line)
    
        image = {}
        for line in f:
            l = line.strip().split()
            image[l[0]] = l[1:4]
        
        return image
    return None

def pos_treebank2wordnet(treebank_tag):
    '''Map the Penn Treebank POS tags to WordNet tags.'''
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        # default
        return nltk.corpus.wordnet.NOUN

def get_mpqa_lexicon(path_to_lexicon=None):
    if path_to_lexicon is not None:
        path = path_to_lexicon
    else:
        path = os.path.abspath('/p/spoclab/tools/CORE/lib/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff')

    with open(path) as f:
        lines = f.readlines()
    words = [re.match('word1=(.*)', line.split()[2]).groups()[0] for line in lines]
    types = [re.match('type=(.*)subj', line.split()[0]).groups()[0] for line in lines]
    polarities = [re.search('polarity=(.*)', line).groups()[0] for line in lines]

    return [words, types, polarities]

def numsyllables(word,prondict):
    numsyllables_pronlist = lambda l: len(list(filter(lambda s: str.isdigit(str(s.encode('ascii', 'ignore')).lower()[-1]), l)))
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]
