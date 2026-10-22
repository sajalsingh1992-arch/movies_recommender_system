
import streamlit as st
import pickle
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fetch_poster(movie_id):
    api_key = st.secrets["TMDB_API_KEY"]

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {"api_key": api_key}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

    except requests.RequestException:
        pass

    return None


# Load movie data
with open("movies.pkl", "rb") as file:
    movies_dict = pickle.load(file)

movies = pd.DataFrame(movies_dict)

# Build text features once, instead of rebuilding them on every click
if "tags" not in movies.columns:
    st.error(
        "The tags column is missing from movies.pkl. "
        "Make sure your original movie dataset includes tags."
    )
    st.stop()

movies["tags"] = movies["tags"].fillna("").astype(str)


@st.cache_resource
def build_count_matrix(tags):
    cv = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )
    count_matrix = cv.fit_transform(tags)
    return count_matrix


count_matrix = build_count_matrix(
    tuple(movies["tags"].tolist())
)


def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]

    # Compare only the selected movie against all movies.
    # This avoids creating the full, memory-heavy similarity matrix.
    distances = cosine_similarity(
        count_matrix[movie_index],
        count_matrix
    ).flatten()

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)

        # Fetch posters from API
        recommended_movies_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movies, recommended_movies_posters


st.title("Movie Recommender System")

selected_movie_name = st.selectbox(
    "Which movie do you like?",
    movies["title"].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)

    col1, col2, col3, col4, col5 = st.columns(5)


    for col, name, poster in zip(
        [col1, col2, col3, col4, col5],
        names,
        posters
    ):
        with col:
            st.header(name)

            if poster:
                st.image(poster)
            else:
                st.caption("Poster unavailable")
