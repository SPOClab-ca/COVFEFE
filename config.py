import os
import configparser

CONFIG_FILE = "config.ini"

_from_cfg_file = None

DEFAULT_NOT_SET = "dasklhglaskjdlkasj"

if os.path.isfile(CONFIG_FILE):
    cfg = configparser.ConfigParser()
    # Prevent configparser from converting keys to lowercase
    cfg.optionxform = str
    cfg.read(CONFIG_FILE)
    _from_cfg_file = cfg._sections["deps"]

def _get_var(key, default=DEFAULT_NOT_SET):
    try:
        if key in os.environ:
            return os.environ[key]
        elif key in _from_cfg_file:
            return _from_cfg_file[key]
    except:
        if default == DEFAULT_NOT_SET:
            raise Exception("Key %s not found in environment or config file %s " % (key, CONFIG_FILE))
        else:
            return default

OPENSMILE_HOME = _get_var("OPENSMILE_HOME")

# Paths to external libraries and relevant files for lexicosyntactic features
stanford_pos_path = _get_var("stanford_pos_path")
stanford_parser_path = _get_var("stanford_parser_path")
lu_analyzer_path = _get_var("lu_analyzer_path")
cfg_rules_path = _get_var("cfg_rules_path")
path_to_dictionary = _get_var("path_to_dictionary")
path_to_freq_norms = _get_var("path_to_freq_norms")
path_to_image_norms = _get_var("path_to_image_norms")
path_to_anew = _get_var("path_to_anew", None)
path_to_warringer = _get_var("path_to_warringer")
path_to_mpqa_lexicon = _get_var("path_to_mpqa_lexicon")
path_to_rst_python = _get_var("path_to_rst_python", None)
path_to_rst = _get_var("path_to_rst", None)
path_to_stanford_cp = _get_var("path_to_stanford_cp")
path_to_lda_model = _get_var("path_to_lda_model")
path_to_lda_wordids = _get_var("path_to_lda_wordids")


nltk_data = _get_var("NLTK_DATA", None)
if nltk_data:
    import nltk
    nltk.data.path.append(nltk_data)
