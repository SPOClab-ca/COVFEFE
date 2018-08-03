#!/usr/bin/env bash

DEP_ROOT=$1
COVFEFE_DIR=$(pwd)
OUT_FILE=env.sh

mkdir -p "$DEP_ROOT"
cd "$DEP_ROOT"

read -p "What is the path to the openSMILE source? " opensmile_home
if [ ! -d "$opensmile_home" ]
then
    echo "$opensmile_home does not exist"
    exit
fi

# Stanford pos tagger
stanford_pos_path=$DEP_ROOT/stanford-postagger-2012-01-06
if [ ! -d "$stanford_pos_path" ]
then
    echo "Stanford pos tagger not found. Downloading"
    wget https://nlp.stanford.edu/software/stanford-postagger-2012-01-06.tgz
    tar -xvzf stanford-postagger-2012-01-06.tgz
fi

stanford_parser_path=$DEP_ROOT/stanford-parser-full-2013-06-20
if [ ! -d "$stanford_parser_path" ]
then
    echo "Stanford parser not found. Downloading"
    wget https://nlp.stanford.edu/software/stanford-parser-full-2013-06-20.zip
    unzip stanford-parser-full-2013-06-20.zip
fi

lu_analyzer_path=$DEP_ROOT/L2SCA-2016-06-30
if [ ! -d "$lu_analyzer_path" ]
then
    echo "LU analyzer not found. Downloading"
    wget http://www.personal.psu.edu/xxl13/downloads/L2SCA-2016-06-30.tgz
    tar -xzf L2SCA-2016-06-30.tgz
fi

path_to_warringer=$DEP_ROOT/Ratings_Warriner_et_al.csv
if [ ! -f "$path_to_warringer" ]
then
    echo "Ratings Warringer not found. Downloading"
    wget https://raw.githubusercontent.com/hwalicki/Twitter-Sentiment-Analysis/master/Ratings_Warriner_et_al.csv
fi


path_to_mpqa_lexicon=$DEP_ROOT/subjclueslen1-HLTEMNLP05.tff
if [ ! -f "$path_to_mpqa_lexicon" ]
then
    echo "Ratings mpqa lexicon not found. Downloading"
    wget https://github.uconn.edu/raw/job13011/BigData/master/subjclueslen1-HLTEMNLP05.tff
fi

path_to_stanford_cp=$DEP_ROOT/stanford-corenlp-full-2016-10-31
if [ ! -d "$path_to_stanford_cp" ]
then
    echo "Stanford CoreNLP not found. Downloading"
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    unzip stanford-corenlp-full-2016-10-31.zip
    cp $COVFEFE_DIR/scripts/lexparser_online.sh stanford-corenlp-full-2016-10-31/
    cp $COVFEFE_DIR/scripts/lexparser_dep.sh stanford-corenlp-full-2016-10-31/
fi

cfg_rules_path=$DEP_ROOT/top_rules.txt
path_to_dictionary=$DEP_ROOT/american-english
path_to_freq_norms=$DEP_ROOT/frequencies.txt
path_to_image_norms=$DEP_ROOT/image.txt
path_to_lda_model=$DEP_ROOT/lda_model_wiki
path_to_lda_wordids=$DEP_ROOT/lda_wordids.txt.bz2

if [ ! -f "$cfg_rules_path" ]
then
    echo "Downloading remaining files"
    wget www.cs.toronto.edu/~complingweb/tools/top_rules.txt
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

echo "export stanford_pos_path=$stanford_pos_path" > "$OUT_FILE"
echo "export stanford_parser_path=$stanford_parser_path" >> "$OUT_FILE"
echo "export lu_analyzer_path=$lu_analyzer_path" >> "$OUT_FILE"
echo "export path_to_warringer=$path_to_warringer" >> "$OUT_FILE"
echo "export path_to_mpqa_lexicon=$path_to_mpqa_lexicon" >> "$OUT_FILE"
echo "export path_to_stanford_cp=$path_to_stanford_cp/" >> "$OUT_FILE"

echo "export cfg_rules_path=$cfg_rules_path" >> "$OUT_FILE"
echo "export path_to_dictionary=$path_to_dictionary" >> "$OUT_FILE"
echo "export path_to_freq_norms=$path_to_freq_norms" >> "$OUT_FILE"
echo "export path_to_image_norms=$path_to_image_norms" >> "$OUT_FILE"
echo "export path_to_lda_model=$path_to_lda_model" >> "$OUT_FILE"
echo "export path_to_lda_wordids=$path_to_lda_wordids" >> "$OUT_FILE"

echo "export OPENSMILE_HOME=$opensmile_home" >> $OUT_FILE

echo "Done. All dependencies saved in $DEP_ROOT"
echo "Environment variables saved in $OUT_FILE"
echo "Remember to source this file before running covfefe"