
import codecs
import re
import networkx as nx
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
import time

class processMethods:

    def __init__(self):
        self = self
        self.TermDict = {}


    def lemmatise_corpus(self, text):
        lemmatiser = WordNetLemmatizer()
        termArray  = []

        for word in text.split():
            # lemmatise the term
            word = lemmatiser.lemmatize(word.lower(), pos='v')
            # remove plural instance
            word = lemmatiser.lemmatize(word)
            termArray.append(word)
        return " ".join(termArray)

    def cleanData(self, line):
        # remove urls
        pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        line = replacedString = re.sub(pattern, '', line, flags=0)
        # remove punctuation
        line = re.sub('[^a-zA-Z0-9]', ' ' , line)
        # break strings into an array
        #line = line.split()
        # lemmatise data
        #print("lemmatising text .... ")
        #line = self.lemmatise_corpus(line)
        # remove stopwords
        #print(line)

        line = [x.lower() for x in line.split() if x not in stop and len(x) > 1]

        return " ".join(line)



    def extractContent(self, path):
        with codecs.open(path, 'r', encoding='utf8', errors="ignore") as file:
                lines = file.read()
        return lines

    def extractCorpusGraph(self, df):
        # doc comes in as unprocessed text
        print(len(doc))
        # break into individual lines

        g = self.plotCorpusToDiGraph(df, "corpusGraph")
        print(len(g))
        print(g['graduate'])



    def plotCorpusToDiGraph(self, corpus, title , graph = nx.DiGraph(), failSafe = False, level = 2):
            try:
                if (failSafe):
                    # purposely crash try/except to force graph rebuild
                    x = 1/0
                print("checking if graph exists ....")
                graph = nx.read_gpickle(title)
            except:
                print("FailSafe --> Creating New Graph")
                print(len(corpus))
                for doc in corpus:
                    doc = doc.split(" ")
                    graph , _ = self.plotArray(doc, level, graph)

                nx.write_gpickle(graph, title)

            return graph

    def plotCorpusToDiGraph2(self, corpus, title , graph = nx.DiGraph(), failSafe = True, level = 2):
            try:
                if (failSafe):
                    # purposely crash try/except to force graph rebuild
                    x = 1/0
                print("checking if graph exists ....")
                graph = nx.read_gpickle(title)
            except:
                print("FailSafe --> Creating New Graph")
                print(len(corpus))
                for doc in corpus:
                    #print(doc)
                    lineArray = doc.split(". ")
                    #print(len(lineArray))
                    print(type(lineArray))
                #    print(lineArray)
                    for line in lineArray:
                        c = self.cleanData(line, stopwords = False)
                        if(len(c) > 1):
                            graph , _ = self.plotArray(c, level, graph)
                            # store graph for faster reload
                nx.write_gpickle(graph, title)

            return graph

        # plotting vectors to the array
    def plotArray(self, array, depth, g):
        counter = 1
        moveCounter = 1
        limit = len(array)
        for a in array[:10]:
            g.add_node(a)
            dummyDepth = depth
            # check that the point does not overrun the array
            if(counter + depth <= limit):
                # forage forward untill the maximum extent of the pointer is reached
                while(dummyDepth != 0):
                    # check if weight already exists and update || create
                    if g.has_edge(a, array[moveCounter]):
                        g[a][array[moveCounter]]['weight'] +=1
                    else:
                        g.add_edge(a, array[moveCounter], weight= 1)
                    # increment counters and reset depth

                    dummyDepth = dummyDepth - 1
                    moveCounter = moveCounter + 1
                counter = counter + 1
                moveCounter = counter
                dummyDepth = depth
            else:
                # chop array to facilitate recursion
                array = array[counter - 1:]
        return g , array

    def calculate_prob_a_given_b(self, A, B, AB):
        likelihood = AB*A/float(B)
        return likelihood
