# In [1]:
import numpy as np
import pandas as pd
import math
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

# In [2]:
dfs = []
for year in range(1940, 2018):
    dfs.append(pd.read_csv('scraped_movies/top_movies_of_%d.csv' % year, encoding = 'cp1252'))
movie_data = pd.concat(dfs)

# In [3]:
dfs = []
for year in range(1940, 2018):
    dfs.append(pd.read_csv('scraped_movies/keywords_for_top_movies_of_%d.csv' % year, encoding = 'cp1252'))
keywords = pd.concat(dfs)

# In [4]:
movie_data.index = range(len(movie_data))
keywords.index = range(len(keywords))

# In [5]:
movie_data.head()

# In [6]:
keywords.head()

# In [7]:
marvel_lookup = keywords.keywords.fillna('').str.contains('marvel-cinematic-universe')

# In [8]:
title_lookup = pd.Series(movie_data.title)
title_lookup.index = movie_data.IMDbId
title_lookup = title_lookup.to_dict()

# In [9]:
furiouses = ['The Fast and the Furious (2001)',
'2 Fast 2 Furious (2003)',
'The Fast and the Furious: Tokyo Drift (2006)',
'Fast & Furious (2009)',
'Fast Five (2011)',
'Furious 6 (2013)',
'Furious Seven (2015)',
'The Fate of the Furious (2017)']

furious_lookup = keywords.IMDbId.map(title_lookup).isin(furiouses)

# In [10]:
princess_lookup = keywords.keywords.fillna('').str.contains('disney-princess')

# In [11]:
jaws_lookup = keywords.IMDbId.map(title_lookup).apply(lambda x: 'jaws' in x.lower())

# In [12]:
decade_lookup = pd.DataFrame(movie_data.release_year.apply(lambda x: math.floor(x/10)*10))
decade_lookup.index = movie_data.IMDbId
decade_lookup = decade_lookup.to_dict()['release_year']

# In [13]:
chocula = CountVectorizer(tokenizer = lambda x: x.split(', '))
genres = chocula.fit_transform(movie_data.genre_list.fillna('xxx')).toarray()
genre_lookup = pd.DataFrame(genres, columns = chocula.get_feature_names())
genre_lookup.index = movie_data.IMDbId

# In [14]:
rank_lookup = pd.Series(movie_data.box_office_rank)
rank_lookup.index = movie_data.IMDbId
rank_lookup = rank_lookup.to_dict()

# In [15]:
sample = keywords[(keywords.IMDbId.map(rank_lookup) < 10)& # Make sure they're movies someone has heard of
                  (keywords.IMDbId.map(title_lookup).apply(len) < 25)  # Make sure the title's aren't too long to display nicely
                 ].sample(10)

pd.DataFrame([sample.IMDbId.map(title_lookup), 
              sample.keywords.apply(lambda x: ', '.join(str(x).split('|')))]).transpose()


# In [16]:
vlad = CountVectorizer(tokenizer = lambda x: x.split('|'), min_df = 0)

megatron = TfidfTransformer()

sparse = vlad.fit_transform(pd.Series(keywords.keywords.fillna('').values))
sample_sparse = vlad.fit_transform(pd.Series(sample.keywords.fillna('').values))


# In [17]:
ns = random.sample(range(len(vlad.get_feature_names())), 5)

# In [18]:
pd.DataFrame(sample_sparse.toarray()[:,ns], index=sample.IMDbId.map(title_lookup), columns=[vlad.get_feature_names()[i] for i in ns])


# In [19]:
weighted = megatron.fit_transform(sample_sparse)

pd.DataFrame(weighted.toarray()[:,ns], index=sample.IMDbId.map(title_lookup), columns=[vlad.get_feature_names()[i] for i in ns]).apply(round, args=(2,))

# In [20]:
# Cheeky wee method for shading the dataframe nicely. Pretty sure I nicked this offa StackOverflow.
def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
    rng = M - m
    norm = colors.Normalize(m - (rng * low),
                            M + (rng * high))
    normed = norm(s.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    return ['background-color: %s' % color for color in c]

# In [21]:
weighted = megatron.fit_transform(sparse)
the_matrix = weighted.dot(weighted.T)
the_matrix = the_matrix[:,sample.index][sample.index,:]
dot = pd.DataFrame(the_matrix.toarray(), index=sample.IMDbId.map(title_lookup), columns=sample.IMDbId.map(title_lookup)).apply(round, args=(2,))
dot.style.apply(background_gradient,
               cmap=sns.light_palette("grey", as_cmap=True),
               m=dot.min().min(),
               M=0.05,
               low=0)

# In [22]:
shrinky = NMF(n_components = 2)

shrunk_sample = shrinky.fit_transform(the_matrix.toarray())

# In [23]:
reduced = pd.DataFrame(shrunk_sample, index=sample.IMDbId.map(title_lookup)).apply(round, args=(2,))

reduced.style.apply(background_gradient,
               cmap=sns.light_palette("grey", as_cmap=True),
               m=reduced.min().min(),
               M=reduced.max().max(),
               low=0)

# In [24]:
# Throwing the whole process into a little method
def make_matrix(df, countvectoriser): 
    megatron = TfidfTransformer()
    sparse = countvectoriser.fit_transform(pd.Series(df.keywords.fillna('').values))
    weighted = megatron.fit_transform(sparse)    
    matrix = weighted.dot(weighted.T)
    movies = pd.Series(countvectoriser.get_feature_names())
    return matrix, movies

# In [25]:
vlad = CountVectorizer(tokenizer = lambda x: x.split('|'), min_df = 10)
matrix, words = make_matrix(keywords, vlad)

# In [26]:
shrinky = NMF(n_components = 100)

shrunk_100 = shrinky.fit_transform(matrix.toarray())

# In [27]:
reduced = pd.DataFrame(shrunk_100[sample.index,:15], index=sample.IMDbId.map(title_lookup)).apply(round, args=(2,))
reduced['...'] = pd.Series(['']*10, index = sample.IMDbId.map(title_lookup))
reduced.style.apply(background_gradient,
                    subset=pd.IndexSlice[:,range(0, 15)],
               cmap=sns.light_palette("grey", as_cmap=True),
               m=reduced[[i for i in range(0,10)]].min().min(),
               M=reduced[[i for i in range(0,10)]].max().max(),
               )

# In [28]:
movie_one = list(keywords.IMDbId.map(title_lookup)).index("Saw (2004)")

# In [29]:
movie_two = list(keywords.IMDbId.map(title_lookup)).index("Chef (2014)")

# In [30]:
avg_movie = shrunk_100.mean(axis=0).reshape(1, -1)

# In [31]:
targets = [movie_one, movie_two] #, 4225]

# In [32]:
target =  shrunk_100[movie_one].reshape(1, -1) + shrunk_100[movie_two].reshape(1, -1)

# In [33]:
best_list = [i for i in np.argsort(cosine_similarity(target, shrunk_100))[0][::-1] if i not in targets][:10]

[keywords.IMDbId.map(title_lookup)[i] for i in best_list]

# In [34]:
plotting_matrix = matrix

similar_matrix = plotting_matrix[jaws_lookup[jaws_lookup == True].index]

axis_1_title = 'Die Hard (1988)'
axis_2_title = "Psycho (1960)"

axis_1_movie = list(keywords.IMDbId.map(title_lookup)).index(axis_1_title)

axis_2_movie = list(keywords.IMDbId.map(title_lookup)).index(axis_2_title)

# In [35]:
titles = keywords[jaws_lookup].IMDbId.map(title_lookup)

# In [36]:
titles

# In [37]:
axis_1 = cosine_similarity(plotting_matrix[axis_1_movie], similar_matrix)[0]
axis_2 = cosine_similarity(plotting_matrix[axis_2_movie], similar_matrix)[0]

# In [38]:
# An extremely janky method for stopping the titles from overlapping each other.
def avoid_overlap(axis_1, 
                  axis_2,
                  x_tolerance = 0.05,
                  y_tolerance = 0.02,
                  increment = 0.01):
    fixed = []
    for x, y in zip(axis_1, axis_2):
        Xs = pd.Series([i[0] for i in fixed])
        Ys = pd.Series([i[1] for i in fixed])
        while ((Xs < x+x_tolerance) & (Xs > x-x_tolerance) & (Ys < y+y_tolerance) & (Ys > y-y_tolerance)).any():
            y += y_tolerance
        fixed.append((x, y))
    return fixed

# In [39]:
pd.DataFrame(list(zip(axis_1, axis_2))).plot(kind='scatter', x=0, y=1, c='w')

# Cause I'm lazy, you gotta fiddle with the values to get the titles to show up in the right spots.
for label, (x, y) in zip(list(titles.values), avoid_overlap(axis_1, 
                                                      axis_2, 
                                                      y_tolerance = 0.0006,
                                                      increment = 0.00002,
                                                     x_tolerance = 1)):
    label = label[:label.find('(')-1]
    plt.annotate(label,
                 fontsize=10,
                 fontname='Garamond',
                xy=(x - (len(label)*0.0007), y + 0.0075))
plt.xlabel(axis_1_title, fontname='Garamond', fontsize = 14)
plt.ylabel(axis_2_title, fontname='Garamond', fontsize = 14)  
plt.show()
