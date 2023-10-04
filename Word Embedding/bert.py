from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Define input tweets
tweets = ["Just watched a great movie", "I love pizza", "This weather is terrible", "I am feeling happy today", "Can't wait for the weekend"]

# Tokenize the tweets and flatten the list of tokens
tokenized_tweets = [tokenizer.tokenize(tweet) for tweet in tweets]
tokens = [token for tweet in tokenized_tweets for token in tweet]

# Compute the unique tokens
unique_tokens = set(tokens)

# Compute the BERT embeddings for the unique tokens
with torch.no_grad():
    embeddings = model(torch.tensor(tokenizer.encode(list(unique_tokens), add_special_tokens=True)))[0]

# Compute the similarity between each unique token and "movie"
movie_token = tokenizer.encode("movie", add_special_tokens=True)
movie_embedding = model(torch.tensor(movie_token).unsqueeze(0))[0][0]
similarity = np.dot(embeddings, movie_embedding.cpu().numpy()) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(movie_embedding.cpu().numpy()))

# Get the top 5 similar tokens to "movie"
similar_tokens = np.argsort(similarity)[-6:-1]
similar_words = [tokenizer.decode(embeddings[i].argmax().item()) for i in similar_tokens]
print(similar_words)

'''
# Import necessary libraries
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Define the dataset of tweets
tweets = ["Just watched a great movie", "I love pizza", "This weather is terrible", "I am feeling happy today", "Can't wait for the weekend"]

# Tokenize the tweets
tokenized_tweets = [word_tokenize(tweet.lower()) for tweet in tweets]

# Train the Word2Vec model on the tokenized tweets
model = Word2Vec(tokenized_tweets, min_count=1)

# Define the word to find similar words for
word = "movie"

# Get the most similar words to the given word
similar_words = [w for w, s in model.wv.most_similar(word, topn=3)]

# Print the similar words
print(f"Similar words to '{word}': {set(similar_words)}")

'''