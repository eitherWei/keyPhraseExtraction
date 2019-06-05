from methods_main2 import *
import time
import pandas as pd
from collections import Counter
import sys
start = time.time()
import nltk
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


####################################################################



#posFriendlyCorpus = dataset.procesedString.apply(extractPosTags)


term_list = []
term_idf_list = []
doc_id_list = []

def buildAndProcessText(targetAmount, corpus):

    for indexVal in range(targetAmount):

        text = methods.extractPosTags(corpus[indexVal])
        #print(text)
        Text = []
        for value in text:
            sentArray = []
            for v in value:
                if v != "_":
                    sentArray.append(v)
            if len(sentArray) > 0:
                Text.append(sentArray)
        print('this run is {}'.format(indexVal))



        graph = methods.plotDiGraph([Text])

        textRankDict = methods.computePageRank(graph)


        #text = all_corpus_rejoined[indexVal]

        # extract all candidate phrases
        all_Phrase = []
        for array in text:
            all_Phrase.extend(array)

        # reduct that to unique in stances
        all_Phrase = list(set(all_Phrase))

        # iterate over and
        for phrase in all_Phrase:
            phraseList = phrase.split()
            if len(phraseList) > 1:
                value = 0
                for p in phraseList:
                    if p in textRankDict:
                        value += textRankDict[p]
                value = value/len(phraseList)
                textRankDict[phrase] = value



        local_id_list = [ indexVal for x in range(len(textRankDict))]
        local_term_list = list(textRankDict.keys())
        local_term_idf_list = list(textRankDict.values())


        term_list.extend(local_term_list)
        term_idf_list.extend(local_term_idf_list)
        doc_id_list.extend(local_id_list)

targetAmount = 211
buildAndProcessText(targetAmount, dataset['procesedString'])

    #print(term_list)
df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})

indexList = methods.extractIndexLocationForAllTargetTerms(df, dataset, targetAmount, title = "indexListDf.pkl", failSafe = True)
indexValues = []
for dict1 in indexList:
    indexValues.append(dict1.values())

print("size of index array {}".format(len(indexValues)))
relIndexLoc = methods.rankLocationIndex(indexValues)
methods.plotIndexResults( relIndexLoc)


#targetAmount = 1
#buildAndProcessText(targetAmount, dataset['procesedString'])

print(10*"-*-")
print((time.time() - start)/60)
