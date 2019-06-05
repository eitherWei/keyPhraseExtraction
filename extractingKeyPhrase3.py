from methods_main2 import *
import pandas as pd
import nltk

import time
start = time.time()
# import the target files
dataset = pd.DataFrame()
# initialise methods class
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
methods = mainMethods(path)
# extracts the files handles
methods.extractFileNames(path)

dataset = methods.extractFles()
print(dataset.columns)
print()

# extract the target keyPhrases and lemmatise them
dataset['targetTerms'] = methods.extractTargetTerms(dataset)
#brokenCorpus = methods.cleanString(datasetSection)

dataset['brokenCorpus'] = methods.tokeniseCorpus(dataset)
brokenCorpus = methods.cleanString(dataset.brokenCorpus)


all_corpus_rejoined = methods.expandNGram(brokenCorpus)

# extract viable terms
def extractPOS(Text):
    desired_tags = ["JJ", "NNS", "NN", "JJS"]
    text = Text.split()
    pos_tag = nltk.pos_tag(text)

    text = []
    for tuple in pos_tag:
        if tuple[1] in desired_tags:
            if len(tuple[0]) > 1:
                text.append(tuple[0])
        else:
            text.append("_")
    return text


doc_id_list = []
term_list = []
term_idf_list = []

# reflects the number of docs to investigate
# max target = 211
targetAmount = 10

for indexVal in range(targetAmount):

    print('this run is {}'.format(indexVal))
    text = extractPOS(dataset.wholeSections[indexVal])
    Text = []
    for value in text:
        if value != "_":
            Text.append(value)

    # build graph from target docs
    #dict = methods.plotDicTGraph([[Text]])
    graph = methods.plotDiGraph([[Text]])
    print(len(graph.nodes()))

    textRankDict = methods.computePageRank(graph)

    text = all_corpus_rejoined[indexVal]

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




print(term_list)
df = pd.DataFrame({"doc_id_list": doc_id_list, "term_list" : term_list, "term_idf_list": term_idf_list})

indexList = methods.extractIndexLocationForAllTargetTerms(df, dataset, targetAmount, title = "indexListDf.pkl", failSafe = True)
indexValues = []
for dict1 in indexList:
    indexValues.append(dict1.values())

print("size of index array {}".format(len(indexValues)))
relIndexLoc = methods.rankLocationIndex(indexValues)
methods.plotIndexResults( relIndexLoc)


print(10*"-*-")
print((time.time() - start)/60)
