""" This module extracts syntactic features. """

import subprocess
import os
import re
import nltk.tree

from utils import file_utils
from utils.lexicosyntactic import yngve

def get_lu_complexity_features(lu_analyzer_path, transcript_filepath, transcript_filename, output_lu_parse_dir):

    ''' This function extracts Lu complexity features

    Parameters:
    lu_analyzer_path: string, path to Lu Complexity Analyzer.
    transcript_filepath: string, path to transcript from which we want to extract Lu complexity features.
    transcript_filename: string, name of transcript file.
    output_lu_parse_dir: string, path to directory that will store parse trees produced.

    Returns:
    lu_keys: list of strings, names of extracted features.
    lu_features_dict: dictionary, mapping feature name to feature value.
    '''

    lu_keys = []
    lu_features_dict = {}

    # Run the Lu complexity analyzer script which produces a csv file with the feature values,
    # and a CFG parse of the text.
    subprocess.call(['python2', 'analyzeText.py',
                     os.path.abspath(transcript_filepath), os.path.abspath(os.path.join(output_lu_parse_dir, transcript_filename))],
                    cwd=lu_analyzer_path)

    # Read the features into the dictionary.
    with open(os.path.join(output_lu_parse_dir, transcript_filename), 'r') as fin_lu:
        headers = fin_lu.readline().strip().split(',')
        lu_features = fin_lu.readline().strip().split(',')
        for i in range(1, len(headers)):
            lu_features_dict[headers[i]] = float(lu_features[i])
            lu_keys += [headers[i]]
    return lu_keys, lu_features_dict

def get_parsetree_features(parser_path, cfg_rules_path, transcript_filepath, transcript_filename, output_parse_dir):

    ''' This function extracts parsetree features.

    Parameters:
    parser_path: string, path to Stanford lexical parser.
    cfg_rules_path: string, path to file containing CFG rules to be extracted.
    transcript_filepath: string, path to transcript from which we want to extract Lu complexity features.
    transcript_filename: string, name of transcript file.
    output_parse_dir: string, string, path to directory that will store parse trees produced.

    Returns:
    parsetree_keys: list of strings, names of extracted features.
    parsetree_features: dictionary, mapping feature name to feature value.
    '''

    parsetree_keys = []
    parsetree_features = {}

    # Build stanford CFG and dependency parses, only if they don't exist already
    target_parse = os.path.join(output_parse_dir, transcript_filename + '.parse')
    if not os.path.exists(target_parse):
        # "oneline" parse (parse for one utterance per line)
        with open(target_parse, 'w') as fout:
            subprocess.call([os.path.join(parser_path, 'lexparser_oneline.sh'), transcript_filepath], stdout=fout)

    # "penn,typedDependencies" parse
    target_depparse = os.path.join(output_parse_dir, transcript_filename + '.depparse')
    if not os.path.exists(target_depparse):
        with open(target_depparse, 'w') as fout:
            subprocess.call([os.path.join(parser_path, 'lexparser_dep.sh'), transcript_filepath], stdout=fout)

    with open(target_parse, 'r') as fin:
        treelist = fin.readlines()

        maxdepth = 0.0
        totaldepth = 0.0
        meandepth = 0.0
        treeheight = 0.0
        numtrees = 0

        ###from Jed
        # To read in the parse trees into nltk tree objects, expect the 'oneline'
        # format from the stanford parser (one utterance tree per line).
        yc = yngve.Yngve_calculator()
        for utterance_tree in treelist:
            if utterance_tree:
                thistree = utterance_tree # read parsed tree from parser-processed file
                numtrees += 1
                pt = nltk.tree.ParentedTree.fromstring(thistree) #convert to nltk tree format
                st = list(pt.subtrees()) #extract list of all sub trees in tree
                nodelist = []
                for subt in st:
                    nodelist.append(subt.label())  # make list of all node labels for subtrees

                Snodes = nodelist.count('S') + nodelist.count('SQ') + nodelist.count('SINV')#count how many nodes are "S" (clauses)

                # A list of the Yngve depth (int) for each terminal in the tree
                depthlist = yc.make_depth_list(pt, [])  # computes depth, need to pass it an empty list
                depthlist = depthlist[:-1] # the last terminal is a punctuation mark, ignore it
                if depthlist:
                    maxdepth += max(depthlist)
                    totaldepth += sum(depthlist)
                    if len(depthlist) > 0:
                        meandepth += 1.0*sum(depthlist)/len(depthlist)
                treeheight += pt.height()

        if numtrees > 0:
            parsetree_features['maxdepth'] = maxdepth / numtrees # or should it be max overall?
            parsetree_features['totaldepth'] = totaldepth / numtrees
            parsetree_features['meandepth'] = meandepth / numtrees
            parsetree_features['treeheight'] = treeheight / numtrees
        else:
            parsetree_features['maxdepth'] = 0
            parsetree_features['totaldepth'] = 0
            parsetree_features['meandepth'] = 0
            parsetree_features['treeheight'] = 0
        parsetree_keys += ['maxdepth', 'totaldepth', 'meandepth', 'treeheight']

        # CFG MEASURES
        # Count frequency of different CFG constituents, using the
        # constructed parse tree

        totNP = 0
        totVP = 0
        totPP = 0
        lenNP = 0
        lenVP = 0
        lenPP = 0
        total_length = 0
        prod_nonlexical = []

        # List of rules to look for
        with open(cfg_rules_path, 'r') as fin:
            rules = fin.read()
            top_rules = rules.strip().split('\n')

            for utterance_tree in treelist:
                if utterance_tree:
                    # Convert to unicode to prevent errors when there
                    # are non-ascii characters
                    t = nltk.tree.Tree.fromstring(utterance_tree)
                    prods = t.productions()
                    for p in prods:
                        if p.is_nonlexical():
                            prod_nonlexical.append(re.sub(" ", "_", str(p)))

                    # Counting phrase types
                    for st in t.subtrees():
                        if str(st).startswith("(NP"):
                            lenNP += len(st.leaves())
                            totNP += 1
                        elif str(st).startswith("(VP"):
                            lenVP += len(st.leaves())
                            totVP += 1
                        elif str(st).startswith("(PP"):
                            lenPP += len(st.leaves())
                            totPP += 1

                    sent_length = len(t.leaves())
                    total_length += sent_length

            if total_length > 0:
                parsetree_features["PP_type_prop"] = 1.0*lenPP/total_length
                parsetree_features["VP_type_prop"] = 1.0*lenVP/total_length
                parsetree_features["NP_type_prop"] = 1.0*lenNP/total_length

                parsetree_features["PP_type_rate"] = 1.0*totPP/total_length
                parsetree_features["VP_type_rate"] = 1.0*totVP/total_length
                parsetree_features["NP_type_rate"] = 1.0*totNP/total_length
            else:
                parsetree_features["PP_type_prop"] = 0
                parsetree_features["VP_type_prop"] = 0
                parsetree_features["NP_type_prop"] = 0

                parsetree_features["PP_type_rate"] = 0
                parsetree_features["VP_type_rate"] = 0
                parsetree_features["NP_type_rate"] = 0

            try:
                parsetree_features["average_PP_length"] = 1.0*lenPP/totPP
            except:
                parsetree_features["average_PP_length"] = 0
            try:
                parsetree_features["average_VP_length"] = 1.0*lenVP/totVP
            except:
                parsetree_features["average_VP_length"] = 0
            try:
                parsetree_features["average_NP_length"] = 1.0*lenNP/totNP
            except:
                parsetree_features["average_NP_length"] = 0

            parsetree_keys += ['PP_type_prop', 'VP_type_prop', 'NP_type_prop',
                               'PP_type_rate', 'VP_type_rate', 'NP_type_rate',
                               'average_PP_length', 'average_VP_length', 'average_NP_length']

            # Normalize by number of productions
            num_productions = len(prod_nonlexical)
            fdist = nltk.probability.FreqDist(prod_nonlexical)

            for prod_rule in top_rules: # need this to ensure we always get same number of CFG features
                if prod_rule in fdist:
                    parsetree_features[prod_rule] = 1.0 * fdist[prod_rule] / num_productions
                else:
                    parsetree_features[prod_rule] = 0.0
                parsetree_keys += [prod_rule]
        
                
    return parsetree_keys, parsetree_features
