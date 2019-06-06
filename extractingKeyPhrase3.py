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


accronymDict  = methods.extractAccronymnsCorpus(list(dataset['brokenCorpus']))

'''
dataset['brokenCorpus'] = methods.tokeniseCorpus(dataset)
brokenCorpus = methods.cleanString(dataset.brokenCorpus)
print(brokenCorpus [0])
# break being mindful of separators
#dataset['stopWordRemoved'] = dataset.brokenCorpus.apply(methods.stopwordRemoval)
#print(5*("\n"))
# returns an array of corpus : corpus = array of docs with tokenisedArrayStrings
#dataset['procesedString'] = methods.cleanString(list(dataset.stopWordRemoved))
#print(dataset['procesedString'][0])
dataset['procesedString'] = brokenCorpus


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
targetAmount = 211

for indexVal in range(targetAmount):

    print('this run is {}'.format(indexVal))
    #this method is performed on wholeCorpus
    #text = extractPOS(dataset.wholeSections[indexVal])

    # this method looks to keep sentence delminated integrity.
    text = methods.extractPosTags(dataset.procesedString[indexVal])

    #altering loop as text is an array within an array
    Text = []
    for value in text:
        sentText = []
        for v in value:
            if v != "_":
                sentText.append(v)
        if len(sentText) > 0:
            Text.append(sentText)
    #print(Text)
    # build graph from target docs
    #dict = methods.plotDicTGraph([[Text]])
    graph = methods.plotDiGraph([Text])
    #print(len(graph.nodes()))

    textRankDict = methods.computePageRank(graph)

    # combines all adjacent terms
    phrase = ""
    phraseList = []
    for t in text:
        for s in t:
            if s != "_":
                phrase  = phrase + " " + s
            else:
                if len(phrase) > 1:
                    phraseList.append(phrase.strip())
                    phrase = ""
    #print(phraseList)

    # phrase  creation two
    # this text contains the original job
    #text = all_corpus_rejoined[indexVal]

    # extract all candidate phrases
    #<------------ uncommenting to construct textrank style phrases
    all_Phrase = []
    for array in text:
        #print(array)
        all_Phrase.extend(array)


    # ngrams formed from deliminators constraints
    text = dataset['procesedString'][indexVal]
    #phraseList = methods.sentenceConstrainedNgrams(text)
    # iterates over the corpus and extracts all none singletons.
    # reduct that to unique in stances
    all_Phrase = list(set(phraseList))
    #print(all_Phrase)
    # iterate over and
    for phrase in all_Phrase:
        #print(phrase)
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
'''

print(10*"-*-")
print((time.time() - start)/60)
