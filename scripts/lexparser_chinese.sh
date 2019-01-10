#!/usr/bin/env bash
# Similar to lexparser_oneline.sh but for Chinese
# Argument 1: output directory
# Argument 2: input file (txt)

scriptdir=`dirname $0`

java -mx1600m -cp "$scriptdir/*:" edu.stanford.nlp.pipeline.StanfordCoreNLP \
 -props $scriptdir/chinese.properties -outputFormat "text" -outputDirectory $1 -file $2
