# reading in docs from xml
#import xml.etree.ElementTree as et
import pandas as pd
from methods_main2 import *
from methods_main3 import *
from bs4 import BeautifulSoup
import xml
import re
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer

start = time.time()
# df for dataset


dataset = pd.DataFrame()
# initialise methods class
path = "/Users/stephenbradshaw/Documents/codingTest/AutomaticKeyphraseExtraction-master/data/"
methods = mainMethods(path)
# extracts the files handles
methods.extractFileNames(path)

# load up full filepath (xml)
methods.df['fileNames'] = methods.df.handle.apply(methods.extractXMLFiles)
# extract text content
methods.df['files'] = methods.df.fileNames.apply(methods.extractContent)
### assigning sectionDictionary to dataset
### breaks the xml into sections allowing us to compartmentalise analysis
### assigning sectionDictionary to dataset
### breaks the xml into sections allowing us to compartmentalise analysis
dataset['sectionsDict'] = methods.df.files.apply(methods.extractSections)
# extract one section  and play around with breaking sentences
# examine best way to confine extraction to sentences
# further we break sentences by commas reasoning that no phrases will over lap this
# method takes document chunks breaks into sentences based on . , and returns a array of docs further brroken into sentence arrasy

#looking at all docs
# each doc is broken to section, which  is further broken into sentences and then into word arrays

dataset['brokenCorpus'] = methods.tokeniseCorpus(dataset)
dataset['stopWordRemoved'] = dataset.brokenCorpus.apply(methods.stopwordRemoval)

# loops over corpus and pulls out accronyms
accronymDict  = methods.extractAccronymnsCorpus(list(dataset['brokenCorpus']))

# takes in accronym dictionary made above and expands them whereever they appear in text
brokenCorpus_augment_anagram = methods.expandAcronymsInText(dataset, accronymDict)

# after stopword removal and accronym extend
# lemmatise this corpus
#brokenCorpus_augment_anagram = methods.lemmatiseTheCorpus(brokenCorpus_augment_anagram)
#brokenCorpus_augment_anagram = methods.stemTheCorpus(brokenCorpus_augment_anagram)

#do stemming

# extract the target keyPhrases and lemmatise them
#dataset['targetTerms'] = methods.extractTargetTerms(dataset)
# lemmatising the targets
#dataset['targetTerms'] = methods.lemmatiseTargetTerms(list(dataset['targetTerms']))
# stemming the targetTerms
#dataset['targetTerms'] = methods.stemTargetTerms(list(dataset['targetTerms']))

# clean it
brokenCorpus = methods.cleanString(brokenCorpus_augment_anagram)

print(len(brokenCorpus))
print(len(brokenCorpus[0]))
print(brokenCorpus[0])
'''
graph = methods.plotDiGraph(brokenCorpus)

print(10*"==")
print(graph.edges(data = "cousin"))
print(10*"-")
print(graph.edges(data = "distantCousin"))

print(nx.info(graph))

#print(graph.edges(data = True))

all_corpus_rejoined = methods.expandNGram(brokenCorpus)
#print(all_corpus_rejoined[0])
# pass to the tfidf extractor , mindful that it is already tokenised


allSent = []
for i in range(len(all_corpus_rejoined)):
    oneSent = []
    for sent in  all_corpus_rejoined[i]:
        oneSent.extend(sent)
    allSent.append(oneSent)
#print(allSent[0])
#print(len(allSent))

def countOccurrences():
    for i in range(len(allSent)):
        for term in dataset.targetTerms[i]:
                if term in allSent[i]:
                    print(term)
                    present = present + 1
                else:
                    absent = absent + 1
        print(10*"-+-")

#print(present)
#print(absent)


def returnTfidfonPreprocessedText(docArray):
    def dummy_fun(doc):
        return doc

    tfidf = TfidfVectorizer(analyzer = 'word', tokenizer = dummy_fun, preprocessor = dummy_fun, token_pattern = None)
    matrix = tfidf.fit_transform(docArray)
    #print(tfidf.vocabulary_)
    #print(matrix.shape)
    #print(docArray)
    return tfidf, matrix

# call the above function
tfidf, matrix = returnTfidfonPreprocessedText(allSent)
# link term with idf values
df = methods.ExtractSalientTerms(tfidf, matrix, title ="tfidf_clean.pkl", failSafe = True)

indexList = methods.extractIndexLocationForAllTargetTerms(df, dataset, title = "indexListDf.pkl", failSafe = True)
indexValues = []
for dict1 in indexList:
    indexValues.append(dict1.values())

print("size of index array {}".format(len(indexValues)))
relIndexLoc = methods.rankLocationIndex(indexValues)
print(relIndexLoc)
methods.plotIndexResults( relIndexLoc)
'''
print(10*"--**--")
print((time.time() - start)/60)
##[132, 273, 251, 87, 318, 205] - standard
#[144, 289, 244, 88, 312, 189]  - sentence separate
#[145, 288, 244, 89, 311, 189] - sentence separate /n removal
#[143, 294, 244, 87, 309, 189] - sent / newline / comma
#[146, 292, 243, 88, 308, 189] - sent / newline / comma / collon
#[146, 293, 242, 89, 307, 189] - sent / newline / comma / collon / semicollon
#[152, 286, 247, 86, 306, 189] - sent / newline / comma / collon / semicollon / acronym resolution
#[208, 297, 237, 79, 254, 191] - sent / newline / comma / collon / semicollon / acronym resolution / stopWord Removal
#[236, 299, 226, 73, 248, 183] - sent / newline / comma / collon / semicollon / acronym resolution / stopWord Removal / Lemmatising
#[259, 322, 248, 78, 214, 144] - sent / newline / comma / collon / semicollon / acronym resolution / stopWord Removal / Stemming
#[261, 318, 247, 79, 214, 146] - sent / newline / comma / collon / semicollon / acronym resolution / stopWord Removal / Stemming / Lemmatising


#[147, 209, 369, 161, 62, 318] - standard textrank - window_2
#[147, 208, 365, 168, 60, 318] - standard textrank - window_3
#[154, 203, 366, 164, 61, 318] - standard textrank - window_3 - distanceDecay 1/distanceFromTargetTerm
#[133, 109, 198, 142, 472, 212] - standard textrank - window_3 - distanceDecay 1/distanceFromTargetTerm -> using phrases generated above


#[135, 204, 146, 45, 2, 734] - allowances made for sentence deliminators
