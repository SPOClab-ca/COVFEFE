#!/usr/bin/env bash

if [ -z "$1" ]
  then
    echo "Please provide a path to save dependencies"
    exit
fi

DEP_ROOT=$1
COVFEFE_DIR=$(pwd)
OUT_FILE=config.ini

mkdir -p "$DEP_ROOT"
cd "$DEP_ROOT"


if [ -z "$2" ]
  then
    read -p "What is the path to the openSMILE source? " opensmile_home
else
    opensmile_home=$2
fi

if [ ! -d "$opensmile_home" ]
then
    echo "$opensmile_home does not exist"
    exit
fi

# Stanford pos tagger
stanford_pos_path=stanford-postagger-2012-01-06
if [ ! -d $stanford_pos_path ];
then
    echo "Stanford pos tagger not found. Downloading"
    wget https://nlp.stanford.edu/software/stanford-postagger-2012-01-06.tgz
    tar -xvzf stanford-postagger-2012-01-06.tgz
fi

stanford_parser_path=stanford-parser-full-2013-06-20
if [ ! -d "$stanford_parser_path" ]
then
    echo "Stanford parser not found. Downloading"
    wget https://nlp.stanford.edu/software/stanford-parser-full-2013-06-20.zip
    unzip stanford-parser-full-2013-06-20.zip
fi

cp $COVFEFE_DIR/scripts/lexparser_oneline.sh "$stanford_parser_path/"
chmod +x "$stanford_parser_path/lexparser_oneline.sh"
cp $COVFEFE_DIR/scripts/lexparser_dep.sh "$stanford_parser_path/"
chmod +x "$stanford_parser_path/lexparser_dep.sh"

lu_analyzer_path=L2SCA-2016-06-30
if [ ! -d "$lu_analyzer_path" ]
then
    echo "LU analyzer not found. Downloading"
    wget http://www.personal.psu.edu/xxl13/downloads/L2SCA-2016-06-30.tgz
    tar -xzf L2SCA-2016-06-30.tgz
fi

path_to_warringer=Ratings_Warriner_et_al.csv
if [ ! -f "$path_to_warringer" ]
then
    echo "Ratings Warringer not found. Downloading"
    wget https://raw.githubusercontent.com/hwalicki/Twitter-Sentiment-Analysis/master/Ratings_Warriner_et_al.csv
fi


path_to_mpqa_lexicon=subjclueslen1-HLTEMNLP05.tff
if [ ! -f "$path_to_mpqa_lexicon" ]
then
    echo "Ratings mpqa lexicon not found. Downloading"
    wget https://github.uconn.edu/raw/job13011/BigData/master/subjclueslen1-HLTEMNLP05.tff
fi

path_to_stanford_cp=stanford-corenlp-full-2016-10-31
if [ ! -d "$path_to_stanford_cp" ]
then
    echo "Stanford CoreNLP not found. Downloading"
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    unzip stanford-corenlp-full-2016-10-31.zip
fi

cp $COVFEFE_DIR/scripts/lexparser_oneline.sh "$path_to_stanford_cp/"
cp $COVFEFE_DIR/scripts/lexparser_dep.sh "$path_to_stanford_cp/"

cfg_rules_path=top_rules.txt
chinese_cfg_rules_path=top_chinese_cfg.txt
path_to_dictionary=american-english
path_to_freq_norms=frequencies.txt
path_to_image_norms=image.txt
path_to_lda_model=lda_model_wiki
path_to_lda_wordids=lda_wordids.txt.bz2

if [ ! -f "$cfg_rules_path" ]
then
    echo "Downloading remaining files"
    wget www.cs.toronto.edu/~complingweb/tools/top_rules.txt
    wget www.cs.toronto.edu/~complingweb/tools/top_chinese_cfg.txt
    wget www.cs.toronto.edu/~complingweb/tools/american-english
    wget www.cs.toronto.edu/~complingweb/tools/frequencies.txt
    wget www.cs.toronto.edu/~complingweb/tools/image.txt

    wget www.cs.toronto.edu/~complingweb/tools/lda_model_wiki
    wget www.cs.toronto.edu/~complingweb/tools/lda_wordids.txt.bz2
    wget www.cs.toronto.edu/~complingweb/tools/lda_model_wiki.expElogbeta.npy
    wget www.cs.toronto.edu/~complingweb/tools/lda_model_wiki.id2word
    wget www.cs.toronto.edu/~complingweb/tools/lda_model_wiki.state
fi

cd $COVFEFE_DIR

echo "[deps]" > "$OUT_FILE"
echo "stanford_pos_path=$DEP_ROOT/$stanford_pos_path" >> "$OUT_FILE"
echo "stanford_parser_path=$DEP_ROOT/$stanford_parser_path" >> "$OUT_FILE"
echo "lu_analyzer_path=$DEP_ROOT/$lu_analyzer_path" >> "$OUT_FILE"
echo "path_to_warringer=$DEP_ROOT/$path_to_warringer" >> "$OUT_FILE"
echo "path_to_mpqa_lexicon=$DEP_ROOT/$path_to_mpqa_lexicon" >> "$OUT_FILE"
echo "path_to_stanford_cp=$DEP_ROOT/$path_to_stanford_cp/" >> "$OUT_FILE"

echo "cfg_rules_path=$DEP_ROOT/$cfg_rules_path" >> "$OUT_FILE"
echo "chinese_cfg_rules_path=$DEP_ROOT/$chinese_cfg_rules_path" >> "$OUT_FILE"
echo "path_to_dictionary=$DEP_ROOT/$path_to_dictionary" >> "$OUT_FILE"
echo "path_to_freq_norms=$DEP_ROOT/$path_to_freq_norms" >> "$OUT_FILE"
echo "path_to_image_norms=$DEP_ROOT/$path_to_image_norms" >> "$OUT_FILE"
echo "path_to_lda_model=$DEP_ROOT/$path_to_lda_model" >> "$OUT_FILE"
echo "path_to_lda_wordids=$DEP_ROOT/$path_to_lda_wordids" >> "$OUT_FILE"

echo "OPENSMILE_HOME=$opensmile_home" >> $OUT_FILE

echo "Done. All dependencies saved in $DEP_ROOT"
echo "Configuration saved in $OUT_FILE"
