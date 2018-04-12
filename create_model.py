# In [1]:
import numpy as np
import pandas as pd
import math
import random
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import gzip
import time
import re

initYear = 1940
now = datetime.now()
today = now.today()
yearToday = now.today().year

startProcessingTime = time.time()

dfs = []
for year in range(initYear, yearToday):
    dfs.append(pd.read_csv('scraped_movies/top_movies_of_%d.csv' %
                           year, encoding='cp1252'))


def cleanTitle(title):
    matchObject = re.match(r"(.*?) (?:\(I+\) )?\(([0-9]{4})", title)
    return matchObject.group(1)


def addSimpleTitle(df):
    return df.assign(simpleTitle=df.title.apply(cleanTitle))


movie_data = pd.concat(dfs)[["IMDbId", "title", "release_year"]]
movie_data_complete = addSimpleTitle(movie_data)

# In [3]:
dfs = []
for year in range(initYear, yearToday):
    dfs.append(pd.read_csv(
        'scraped_movies/keywords_for_top_movies_of_%d.csv' % year, encoding='cp1252'))
keywords = pd.concat(dfs)

# In [4]:
movie_data.index = range(len(movie_data))
keywords.index = range(len(keywords))

# In [24]:
# Throwing the whole process into a little method


def make_matrix(df, countvectoriser):
    megatron = TfidfTransformer()
    sparse = countvectoriser.fit_transform(
        pd.Series(df.keywords.fillna('').values))
    weighted = megatron.fit_transform(sparse)
    matrix = weighted.dot(weighted.T)
    movies = pd.Series(countvectoriser.get_feature_names())
    return matrix, movies


# In [25]:
vlad = CountVectorizer(tokenizer=lambda x: x.split('|'), min_df=10)
matrix, words = make_matrix(keywords, vlad)

# In [26]:
shrinky = NMF(n_components=100)

shrunk_100 = shrinky.fit_transform(matrix.toarray())


def saveToFileAndCompress(object, filename):
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object))
    file.close()


fileName = "hanibalVectorModel-" + today.strftime("%d-%m-%Y") + ".gz"

indexFileName = "hanibalVectorIndex-" + today.strftime("%d-%m-%Y") + ".gz"

saveToFileAndCompress(shrunk_100, fileName)

saveToFileAndCompress(movie_data_complete, indexFileName)

endProcessingTime = time.time()
elapsedTimeInSeconds = endProcessingTime - startProcessingTime

print("All completed in: %ss" % elapsedTimeInSeconds)
