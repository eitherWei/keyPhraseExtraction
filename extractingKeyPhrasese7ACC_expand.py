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

dataset['brokenCorpus'] = methods.tokeniseCorpus(dataset)

print(dataset.brokenCorpus[0])


# retrieve accronyms
accronymDict  = methods.extractAccronymnsCorpus(list(dataset['brokenCorpus']))
# change the format to suit required format of method
dataset['formatTextForAccExp'] = dataset.brokenCorpus.apply(methods.formatTextForAccExp)
# loop over text and expand on accronyms present
dataset['brokenCorpus_augment_anagram'] = methods.expandAcronymsInText(list(dataset['formatTextForAccExp']), accronymDict)
# clean the corpus
dataset['procesedString'] = methods.cleanString(dataset.brokenCorpus_augment_anagram)


# create all candidate phrase list
all_corpus_rejoined = methods.expandNGram(list(dataset['procesedString']))
# extract terms of interest
dataset['targetTerms'] = methods.extractTargetTerms(dataset)

# create df to hold results
df = pd.DataFrame()
# variable for holding iteration number
targetAmount = 2
# graph and pageRank text
for i in range(targetAmount):
    df_temp = methods.constructTextRankGraph(i, dataset)
    # add row at index b from dataframe dfObj2 to dataframe dfObj
    df = df.append(df_temp, ignore_index=True)

indexList = methods.extractIndexLocationForAllTargetTerms(df, dataset, targetAmount, title = "indexListDf.pkl", failSafe = True)
indexValues = []
for dict1 in indexList:
    indexValues.append(dict1.values())

print("size of index array {}".format(len(indexValues)))
relIndexLoc = methods.rankLocationIndex(indexValues)
methods.plotIndexResults( relIndexLoc)


print(10*"-+-")
print((time.time() - start)/60)
