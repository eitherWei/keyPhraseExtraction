import codecs
import re
import networkx as nx
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
import pandas as pd

import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

from gensim import corpora, models , similarities
import gensim

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
        # line = line.split()
        # lemmatise data
        # print("lemmatising text .... ")
        line = self.lemmatise_corpus(line)
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



    def plotCorpusToDiGraph(self, corpus, title = "stored_graph.pkl", graph = nx.DiGraph(), failSafe = True, level = 2):
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

    def clusteringMethods(self, methods, loadDump = False, title = "d_clusters.pkl", num_clusters = 10):

        # the distance between all of the docs
        dist = 1 - cosine_similarity(methods.tfidf_matrix)

        if(loadDump):
            print("loading stored results")
            km = joblib.load(title)
        else:
            #num_clusters = 10
            km = KMeans(n_clusters = num_clusters)
            clusters = km.fit(methods.tfidf_matrix)
            clusters = km.labels_.tolist()
            joblib.dump(km,  title)

        # load into a dataFrame
        df = pd.DataFrame({"cluster" : clusters})
        print(df.cluster.value_counts())



        methods.df['clusterAssignment'] = km.labels_.tolist()

        print(methods.df.head())

        #print(km.labels_.tolist()[0])



        ########################################################################
        # sorting clusters by ngrams
        print("Top terms per cluster")
        clusterTermDict = {}
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        for i in range(num_clusters):
            print("Cluster %d words:" % i , end=',')

            for ind in order_centroids[i, :6]:
                print(methods.wordStore[ind])
                clusterTermDict[i] = methods.wordStore[ind]


        ########################################################################

        ########################################################################
        '''
        # multidimensional scaling
        MDS()

        mds = MDS(n_components = 2, dissimilarity = "precomputed", random_state =1)

        pos = mds.fit_transform(dist)

        xs, ys = pos[:, 0], pos[:, 1]

        cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}

        mds_df = pd.DataFrame(dict(x = xs, y = ys, label = km.labels_.tolist()))

        groups = mds_df.groupby('label')

        fig, ax = plt.subplots(figsize=(17,9))
        ax.margins(0.05)

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
                label=clusterTermDict[name], color=cluster_colors[name],
                mec='none')
            ax.set_aspect('auto')
            ax.tick_params(\
                axis= 'x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
            ax.tick_params(\
                axis= 'y',         # changes apply to the y-axis
                which='both',      # both major and minor ticks are affected
                left='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelleft='off')

        ax.legend(numpoints=1)  #show legend with only 1 point

        #plt.show() #show the plot

        from gensim.test.utils import common_texts
        from gensim.corpora.dictionary import Dictionary
        from gensim import corpora, models, similarities

        print(methods.df.sanitiseData[0])
        text = methods.df.sanitiseData[0].split()
        text2 = methods.df.sanitiseData[1].split()
        textArray = []
        textArray.append(text)
        textArray.append(text2)

        dictionary = Dictionary(textArray)
        corpus = [dictionary.doc2bow(text) for text in textArray]

        #lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word = dictionary, passes=100)

        print(lda.show_topics())

        '''

        return silhouette_score(methods.tfidf_matrix, km.labels_)

        ########################################################################
