import streamlit as st
import pickle
import pandas as pd
import requests
# The only two required mathematical imports
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]

    url = f"https://themoviedb.org{movie_id}"

    params = {
        "api_key": api_key
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    poster_path = data.get("poster_path")

    if poster_path:
        return "https://tmdb.org" + poster_path

    return None


def recommend(movie):
    # FIXED: Added [0] to extract the pure integer index out of the Pandas object
    movie_index = movies[movies['title'] == movie].index[0]

    # Calculate the similarity matrix using scikit-learn
    cv = CountVectorizer(max_features=5000, stop_words='english')
    count_matrix = cv.fit_transform(movies['tags'])
    similarity = cosine_similarity(count_matrix)


    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True,
                         key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)

        # fetch posters from api
        recommended_movies_posters.append(fetch_poster(movie_id))

    return recommended_movies, recommended_movies_posters


movies_dict = pickle.load(open('movies.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

st.title('Movie Recommender System')

selected_movie_name = st.selectbox(
    'which movie you like ?',
    movies['title'].values)
if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.header(names[0])
        st.image(posters[0])

    with col2:
        st.header(names[1])
        st.image(posters[1])

    with col3:
        st.header(names[2])
        st.image(posters[2])

    with col4:
        st.header(names[3])
        st.image(posters[3])

    with col5:
        st.header(names[4])
        st.image(posters[4])
