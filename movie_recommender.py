import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load datasets
# -----------------------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

# -----------------------------
# Convert genres and keywords
# -----------------------------
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# -----------------------------
# Convert cast (top 3 actors)
# -----------------------------
def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert_cast)

# -----------------------------
# Extract director from crew
# -----------------------------
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# -----------------------------
# Create tags column
# -----------------------------
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Remove spaces inside words
movies['tags'] = movies['tags'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create new dataframe
new_df = movies[['movie_id', 'title', 'tags']].copy()

# Convert list to string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# -----------------------------
# Vectorization
# -----------------------------
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

# -----------------------------
# Cosine Similarity
# -----------------------------
similarity = cosine_similarity(vectors)

# -----------------------------
# Recommendation Function
# -----------------------------
def recommend(movie):
    if movie not in new_df['title'].values:
        print("Movie not found in dataset.")
        return
    
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True,
                        key=lambda x: x[1])[1:6]
    
    print(f"\nMovies similar to {movie}:\n")
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

import pickle

# Save processed dataframe
pickle.dump(new_df, open('movies.pkl', 'wb'))

# Save similarity matrix
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Model saved successfully!")