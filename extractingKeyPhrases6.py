from methods_main2 import *
import time
import pandas as pd
from collections import Counter
import sys
start = time.time()
import nltk

start = time.time()
# import a dataset
dataset = pd.DataFrame()
# initialise methods class
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
methods = mainMethods(path)
# extracts the files handles
methods.extractFileNames(path)

# clump of extraction methods
dataset = methods.extractFles()

#return methods.cleanSent( text)
dataset['brokenCorpus'] = methods.tokeniseCorpus(dataset)

# break being mindful of separators
dataset['stopWordRemoved'] = dataset.brokenCorpus.apply(methods.stopwordRemoval)

# returns an array of corpus : corpus = array of docs with tokenisedArrayStrings
dataset['procesedString'] = methods.cleanString(list(dataset.stopWordRemoved))

# extract the target keyPhrases and lemmatise them
dataset['targetTerms'] = methods.extractTargetTerms(dataset)

#posFriendlyCorpus = dataset.procesedString.apply(methods.extractPosTags)

Text = []
#for text in dataset.procesedString:
graph = methods.plotDiGraph([dataset.procesedString[0]])
#textRankDict = methods.computePageRank(graph)
print(len(graph.nodes()))
inoutDict = {}
for a in list(graph.nodes())[:2]:
    for k, v in graph[a].items():
        print(k , v )
        if k in inoutDict:
            inoutDict[k] += v['cousin']
        else:
            inoutDict[k] = v['cousin']






print((time.time() - start)/60)
