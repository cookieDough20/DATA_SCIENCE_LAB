# Text Analytics and Recommendation System
# 1. Text Preprocessing (Tokenization + Stop-word Removal)
# 2. Content-Based Recommendation System
# Application: Movie Recommendation
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
# ---------------------------------------------------
# TEXT PREPROCESSING
# ---------------------------------------------------
print("----- Text Preprocessing -----")
# Sample movie description
text = "The movie was amazing, exciting, and full of action scenes!"
# Convert to lowercase
text = text.lower()
# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))
# Tokenization
tokens = word_tokenize(text)
# Stop-word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [
    word for word in tokens
    if word not in stop_words
]
print("Original Text:")
print("The movie was amazing, exciting, and full of action scenes!")
print("\nProcessed Tokens:")
print(filtered_tokens)
# ---------------------------------------------------
# CONTENT-BASED RECOMMENDATION SYSTEM
# ---------------------------------------------------
print("\n\n----- Content-Based Recommendation -----")
# Movie dataset
movies = pd.DataFrame({
    'Movie': [
        'Avengers',
        'Batman',
        'Superman',
        'Titanic',
        'Inception'
    ],    
    'Description': [
        'action superhero marvel fight',
        'dark superhero action detective',
        'superhero flying action hero',
        'romance ship love drama',
        'dream science fiction thriller'
    ]
})
print("Movie Dataset:")
print(movies)
# Convert text descriptions into TF-IDF vectors
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['Description'])
# Compute similarity matrix
similarity = cosine_similarity(tfidf_matrix)
# Select movie for recommendation
movie_name = 'Batman'
# Find movie index
movie_index = movies[movies['Movie'] == movie_name].index[0]
# Get similarity scores
scores = list(enumerate(similarity[movie_index]))
# Sort by similarity
sorted_scores = sorted(
    scores,
    key=lambda x: x[1],
    reverse=True
)
print("\nRecommended Movies for Batman:")
# Display top recommendations
for i in sorted_scores[1:4]:
    print(movies.iloc[i[0]]['Movie'])