#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# **Define the datasets**

# In[2]:


Movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
Credits = pd.read_csv('dataset/tmdb_5000_credits.csv')


# **TEST THE CODE BY PROPELING THE FIRST MOVIES FROM THE DATASET**

# In[3]:


Movies.head(1)


# In[4]:


Credits.head(1)


# In[5]:


Credits.head(1)['crew'].values


# In[6]:


Credits.head(1)['cast'].values


# ******ORGANIZE THE DATASET******

# *MERGE THE MOVIES CREDITS TO THE MOVIES DATASET*

# In[7]:


movies = Movies.merge(Credits, on='title')


# In[8]:


Movies.shape


# In[9]:


Credits.shape


# In[10]:


movies.head(1)


# In[ ]:


movies.iloc[0].overview


# In[12]:


movies.iloc[0].production_companies


# In[13]:


movies['original_language'].value_counts()


# **VARIABLES THAT ARE GOING TO HELP US ON THE RECOMMENDATION ALGORITHM**

# **Redefine movies dataset with the only variables we are going to need**

# In[14]:


#genres 
#id
#keywords
#title
#overview
#cast
#crew

movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]


# In[15]:


movies.info()


# In[16]:


movies.head()


# **Check if there's any null variable on all the movies**

# In[17]:


movies.isnull().sum() #no null values


# **Drop movies with null variable(s)**

# In[18]:


movies.dropna(inplace=True)


# In[19]:


movies.isnull().sum()


# **Check if there's any duplication**

# In[20]:


movies.duplicated().sum() #no duplicates


# **Organize some variables**

# In[21]:


movies.iloc[0]['genres']


# **Helper Function to format movies genre**

# In[22]:


#From this format to the one below
#'[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'
#['Action', 'Adventure', 'Fantasy', 'Science Fiction']


# In[23]:


import ast as ast


# In[24]:


def formatGenre(obj):
    List = []
    for item in ast.literal_eval(obj):
        List.append(item['name'])
    return List


# **Organize some variables for the movie**

# In[25]:


movies['genres'] = movies['genres'].apply(formatGenre)


# In[26]:


movies.iloc[0]['genres']


# In[27]:


movies.head()


# **Helper Function to format movies cast**

# In[28]:


movies.iloc[0]['cast']


# In[29]:


#Take just the first 6 actors
#['Sam Worthington', 'Zoe Saldana', 'Sigourney Weaver', 'Stephen Lang', 'Michelle Rodriguez', 'Giovanni Ribisi']
def formatCast(obj):
    List = []
    counter = 0
    for item in ast.literal_eval(obj):
        if counter != 6:
            List.append(item['name'])
            counter += 1
        else:
            break
    return List


# In[30]:


movies['cast'] = movies['cast'].apply(formatCast)


# In[31]:


movies.iloc[0]['cast']


# In[32]:


movies.head()


# **Helper Function to format movies crew**

# In[33]:


movies.iloc[0]['crew']


# In[34]:


#Take just the director
#['James Cameron']
def formatCrew(obj):
    List = []
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            List.append(item['name'])
            break
    return List


# In[35]:


movies['crew'] = movies['crew'].apply(formatCrew)


# In[36]:


movies.head()


# **Helper Function to format movies overview**

# In[37]:


movies.iloc[0]['overview']


# In[38]:


# Let's split the overview into words(using lambda function)
movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[39]:


movies.head()


# 

# In[40]:


movies.iloc[0]['keywords']


# **Helper Function to format movies keywords**

# In[41]:


#Let's get the only name of the keywords
def formatKeywords(obj):
    List = []
    for item in ast.literal_eval(obj):
        List.append(item['name'])
    return List


# In[42]:


movies['keywords'] = movies['keywords'].apply(formatKeywords)


# In[43]:


movies.head()


# Functions to apply while reccomending a movie:
# *Name that contains 1 or more words should be handled
# *Spaces should be handled
# *Empty strings should be handled
# *Remove spaces
# 

# In[44]:


movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])


# In[45]:


movies.head()


# In[46]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[47]:


movies.head()


# In[48]:


new_movies = movies[['movie_id', 'title', 'tags']]


# In[49]:


new_movies.head()


# **Concat the tags into one string**

# In[50]:


new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))


# In[51]:


new_movies.head()


# In[52]:


new_movies['tags'][0]


# **Convert the tags variable into lowercase**

# In[53]:


new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())


# In[54]:


new_movies.head()


# In[55]:


new_movies['tags'][0]


# In[56]:


new_movies['tags'][1]


# **Natural Language Toolkit**
# *The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing for English written in the Python programming language*

# In[57]:


import nltk


# In[58]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[59]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[60]:


ps.stem('loving')


# ***Let's convert a collection of text documents to a matrix of token counts. Using the Count Vectorization - scikit-learn (One Hot-Encoding)***

# In[ ]:


new_movies['tags'] = new_movies['tags'].apply(stem)


# In[61]:


import sklearn


# In[62]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')


# **Convert the movies tags to numpy array**

# In[63]:


tags_vectors = cv.fit_transform(new_movies['tags']).toarray()


# In[64]:


tags_vectors.shape


# In[65]:


tags_vectors[0]


# In[66]:


tag_length = len(cv.get_feature_names())


# In[69]:


from sklearn.metrics.pairwise import cosine_similarity


# In[77]:


similarity = cosine_similarity(tags_vectors)


# In[78]:


similarity.shape


# In[79]:


similarity[0] #compare similarity of the first movie with all the other movies


# **Function to recommend movies according to the similarity**

# In[80]:


#To fetch the index of the movie
new_movies[new_movies['title'] == 'Avatar'].index[0]


# In[82]:


#To fetch the info of the movie
#new_movies.iloc[0]
new_movies[new_movies['title'] == 'Avatar']


# In[ ]:


#To sort the similarity array in descending order for the first movie [0]
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])


# In[84]:


#Get the first 10 movies similar to the first movie
sorted(list(enumerate(similarity[0])), reverse=True, key=lambda x:x[1])[1:11]  


# In[86]:


def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:11]
    for i in movie_list:
        print(new_movies.iloc[i[0]].title)


# In[87]:


recommend('Avatar')

