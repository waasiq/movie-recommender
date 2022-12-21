import numpy as np
import pandas as pd
import ast as ast

import nltk as nltk
from nltk.stem.porter import PorterStemmer

import sklearn as sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

Movies = pd.read_csv('dataset/tmdb_5000_movies.csv')
Credits = pd.read_csv('dataset/tmdb_5000_credits.csv')

#* Merging the movies and credits datasets on bases of title
movies = Movies.merge(Credits, on='title')

#* Only keeping the important columns
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
movies.dropna(inplace=True)

def formatGenre(obj):
    List = []
    for item in ast.literal_eval(obj):
        List.append(item['name'])
    return List

movies['genres'] = movies['genres'].apply(formatGenre)

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

movies['cast'] = movies['cast'].apply(formatCast)

# Just keeping the director's name
def formatCrew(obj):
    List = []
    for item in ast.literal_eval(obj):
        if item['job'] == 'Director':
            List.append(item['name'])
            break
    return List

movies['crew'] = movies['crew'].apply(formatCrew)

# Let's split the overview into list using a lambda function
movies['overview'] = movies['overview'].apply(lambda x:x.split())

#Let's get the only name of the keywords
def formatKeywords(obj):
    List = []
    for item in ast.literal_eval(obj):
        List.append(item['name'])
    return List

movies['keywords'] = movies['keywords'].apply(formatKeywords)

# * Functions to apply while reccomending a movie:
# *Name that contains 1 or more words should be handled
# *Spaces should be handled
# *Empty strings should be handled
# *Remove spaces

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

#* Creating a new column called tags which will contain all the important information
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_movies = movies[['movie_id', 'title', 'tags']]

#* Concatenate all the lists into a string
new_movies['tags'] = new_movies['tags'].apply(lambda x: " ".join(x))

#* Convert the tags into lowercase
new_movies['tags'] = new_movies['tags'].apply(lambda x: x.lower())

#* Stemming the words
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

new_movies['tags'] = new_movies['tags'].apply(stem)

#* Applying Text Vectorization (Bag of Words)
cv = CountVectorizer(max_features=5000, stop_words='english') 
vectorized_movies = cv.fit_transform(new_movies['tags']).toarray()

#* Calculating the cosine similarity
similarity = cosine_similarity(vectorized_movies)

#To sort the similarity array in descending order for the first movie [0]

def recommend(movie):
    movie_index = new_movies[new_movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_movies.iloc[i[0]].title)

recommend('The Dark Knight Rises')

pickle.dump(new_movies, open('movies_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))