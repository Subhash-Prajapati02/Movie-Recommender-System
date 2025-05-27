from django.shortcuts import render
from django.views.generic import TemplateView
import numpy as np 
import pandas as pd 
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
# Create your views here.

class Home(TemplateView):
    template_name = "index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # context[""] = 
        return context
    
    
    def convert(self, obj):
                L = []
                for i in ast.literal_eval(obj):
                    L.append(i['name'])
                return L

    def convert3(self, obj):
                L = []
                ct = 0
                for i in ast.literal_eval(obj):
                    if ct != 3:
                        L.append(i['name'])
                        ct = ct + 1
                    else:
                        break
                return L

    def fetch_director(self, obj):
                L = []
                for i in ast.literal_eval(obj):
                    if i['job'] == 'Director':
                        L.append(i['name'])
                        break
                return L


    def stem(self, text, *args):
            ps = PorterStemmer()
            y = []
            for i in text.split():
                y.append(ps.stem(i))
            return " ".join(y)

    def recommend(self, movie, new_df, similarity):
        try:
            movie_index = new_df[new_df['title'] == movie].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
            return recommended_movies
        except IndexError:
            return [movie + ' not found']

    def get_context_data(self, **kwargs):
            context = super().get_context_data(**kwargs)
            movies = pd.read_csv('recommender/movies.csv')
            credits = pd.read_csv('recommender/credits.csv')
            movies = movies.merge(credits, on='title')
            movies = movies[['genres', 'id', 'keywords', 'title', 'overview', 'cast', 'crew']]
            movies.dropna(inplace=True)
            movies['genres'] = movies['genres'].apply(self.convert)
            movies['keywords'] = movies['keywords'].apply(self.convert)
            movies['cast'] = movies['cast'].apply(self.convert3)
            movies['crew'] = movies['crew'].apply(self.fetch_director)
            movies['overview'] = movies['overview'].apply(lambda x: x.split())
            movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
            movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
            new_df = movies[['id', 'title', 'tags']]
            new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
            new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
            new_df['tags'] = new_df['tags'].apply(lambda x: self.stem(x))
            cv = CountVectorizer(max_features=5000, stop_words='english')
            vectors = cv.fit_transform(new_df['tags']).toarray()
            similarity = cosine_similarity(vectors)

            # Get the movie name from the input
            movie_name = self.request.GET.get('movie_name', '')
            if movie_name:
                context['recommendations'] = self.recommend(movie_name, new_df, similarity)
            else:
                context['recommendations'] = []

            return context

class MovieDetail(TemplateView):
    template_name = 'movie_details.html'

    def fetch_director(self, obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L
    
    def recommend(self, movie, new_df, similarity):
        try:
            movie_index = new_df[new_df['title'] == movie].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
            recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
            return recommended_movies
        except IndexError:
            return [movie + ' not found']

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        movie_title = self.kwargs.get('movie')
        movies = pd.read_csv('recommender/movies.csv')
        credits = pd.read_csv('recommender/credits.csv')
        movies = movies.merge(credits, on='title')

        # Ensure the 'tags' column is created
        if 'tags' not in movies.columns:
            movies['tags'] = movies['overview'] + ' ' + movies['genres'] + ' ' + movies['keywords'] + ' ' + movies['cast'] + ' ' + movies['crew']
        movies['tags'] = movies['tags'].astype(str)  # Ensure all entries are strings
        movies['tags'] = movies['tags'].str.lower()  # Convert to lowercase

        # Get the specific movie
        movie = movies[movies['title'] == movie_title].iloc[0]

        # Prepare the DataFrame for recommendations
        new_df = movies[['id', 'title', 'tags']]
        
        # Create the CountVectorizer and compute similarity
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(new_df['tags']).toarray()
        similarity = cosine_similarity(vectors)

        context['movie'] = {
            'title': movie['title'],
            'release_date': movie['release_date'],
            'director': self.fetch_director(movie['crew']),
            'genres': movie['genres'],
            'overview': movie['overview']
        }
        context['recommendations'] = self.recommend(movie_title, new_df, similarity)

        return context

# Harry Potter and the Half-Blood Prince