import nltk
import os
from utils.lexicosyntactic import functions
from utils.lexicosyntactic import feature

class Utterance(object):
    
    def __init__(self, speaker_id, start_time=None, end_time=None, data={}):
        '''start_time : float, number of seconds since beginning of file where the utterance starts
           end_time : float, number of seconds since beginning of file where the utterance ends
           speaker_id : int, numeric ID of the subject speaking the utterance
           data : dictionary, value=annotationID, key=annotation text'''
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id
        self.data = data
    

class Transcript(object):
    
    def __init__(self, filepath, label, pos_tagger_path=None):
        self.filepath = filepath
        self.filename = functions.get_filename(filepath)
        self.label = label
        
        # Base path for Stanford POS tagger
        if pos_tagger_path is not None:
            self.pos_tagger_path = pos_tagger_path
        else:
            # Default path for POS tagger. Can change when initializing the object.
            self.pos_tagger_path = os.path.abspath("../stanford-postagger-2012-01-06")
        
        # Derive the following attributes from the provided file (stored as strings)
        # Each file is associated with a subject ID (this is the patient with a given 
        # pathology), and a speaker ID (this may be either the patient or a healthy 
        # interviewer).
        # tuple_ids = functions.get_subject_sample(self.filename)
        # self.subject_id, self.session_id = tuple_ids[0], tuple_ids[1]
        # self.speaker_id = self.subject_id
        # if len(tuple_ids) > 2:
        #     self.speaker_id = tuple_ids[2]
        
        # The raw text in unicode format
        self.raw = ""
    
    def __str__(self):
        return "Transcript(%s)" % (self.filename)
    
    def __repr__(self):
        return self.__str__()
    
    
class PlaintextTranscript(Transcript):
    
    utterance_sep = ' . '
    
    def __init__(self, filepath, label, pos_tagger_path=None):
        
        super(PlaintextTranscript, self).__init__(filepath, label, pos_tagger_path=pos_tagger_path)
        
        # A list of strings
        self.utterances = []
        
        # A list of lists of strings (each row is an utterance, and each utterance list
        # is a list of tokens).
        self.tokens = []
        
        if pos_tagger_path is not None:
            # Initialize the POS tagger
            self.pos_tagger = nltk.tag.StanfordPOSTagger(os.path.join(self.pos_tagger_path, "models/english-left3words-distsim.tagger"),
                                                         path_to_jar=os.path.join(self.pos_tagger_path, "stanford-postagger.jar"))
        
        # A list of lists of tuples of (token, POStag), each row is an utterance
        self.tokens_pos_tagged = None
        
        # A FeatureSet object (contains Feature objects) - init as empty
        self.feature_set = feature.FeatureSet(features=[])
        
        with open(filepath, 'r') as transcript_fin:
            # The transcript contains only plaintext (alphanumeric characters, spaces, apostrophes).
            # Utterances are separated by a standard separator token. Everything is lowercased (for 
            # DemBank, need to lowercase the first character of each utterance!).
            self.raw = transcript_fin.read()
            self.utterances = [utt.lower() for utt in self.raw.split(self.utterance_sep) if utt]
            
            for utt in self.utterances:
                self.tokens += [nltk.word_tokenize(utt)]
    
    def add_feature(self, new_feature):
        '''Parameters:
        new_feature : either a Feature object or a list of Feature objects.
        
        Return: nothing.'''
        self.feature_set.add(new_feature)
    
    def get_pos_tagged(self):
        # Generate the POS tags for the tokens in the transcript, if they are not 
        # available already. Use the Stanford POS tagger.
        if self.tokens_pos_tagged is None:
            # list of lists of tuples of (token, POStag) pairs, each row is an utterance
            self.tokens_pos_tagged = [] 
            for utt in self.tokens:
                # Given a list of string tokens, the tagger returns a list of tuples (token, POStag)
                # using the Penn treebank tagset
                self.tokens_pos_tagged += [self.pos_tagger.tag(utt)]
        return self.tokens_pos_tagged

class TranscriptSet(object):
    
    def __init__(self, dataset=[], name='Untitled'):
        self.dataset = dataset
        self.name = name
    
    def append(self, transcript):
        '''Add a new transcription to the dataset.'''
        self.dataset += [transcript]
    
    def get_length(self):
        '''Return the number of transcriptions in the set.'''
        return len(self.dataset)
    
    def __getitem__(self, index):
        '''Overload the [] operator to enable subscription.'''
        return self.dataset[index]
    
    def __str__(self):
        return "TranscriptSet(%s, %d transcripts)" % (self.name, self.get_length())
    
    def __repr__(self):
        return self.__str__()
    
    
    