import os
import nltk

OPENSMILE_HOME = os.environ.get("OPENSMILE_HOME", "/Users/dl/programs/opensmile-2.3.0")

nltk.data.path.append("/pkgs/nltk_data/")

# Paths to external libraries and relevant files for lexicosyntactic features
stanford_pos_path ='/p/spoclab/tools/CORE/lib/stanford-postagger-2012-01-06/'
stanford_parser_path ='/u/yancheva/Documents/grad/code/stanford-parser-full-2013-06-20/'
lu_analyzer_path ='/p/spoclab/tools/CORE/lib/L2SCA-2016-06-30/'
cfg_rules_path ='/u/yancheva/Documents/grad/code/feature_extraction/text/top_rules.txt'
path_to_dictionary ='/u/yancheva/Documents/grad/code/feature_extraction/text/american-english'
path_to_freq_norms ='/u/yancheva/Documents/grad/code/feature_extraction/text/frequencies.txt'
path_to_image_norms ='/u/yancheva/Documents/grad/code/feature_extraction/text/image.txt'
path_to_anew ='/p/spoclab/data/Dictionaries/ANEW2010/ANEW2010All.txt'
path_to_warringer ='/p/spoclab/data/Dictionaries/Warriner/Ratings_Warriner_et_al.csv'
path_to_mpqa_lexicon ='/p/spoclab/tools/CORE/lib/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff'
path_to_rst_python ='/p/spoclab/tools/Discourse/rstenv/bin/python'
path_to_rst ='/p/spoclab/tools/Discourse/RST/src/'
path_to_rst_output =''
path_to_stanford_cp ='/p/spoclab/tools/Stanford/stanford-corenlp-full-2016-10-31/*'
path_to_lda_model ='/p/spoclab/models/LDA/lda_model_wiki'
path_to_lda_wordids ='/p/spoclab/models/LDA/lda_wordids.txt.bz2'