'''
import csv
import os
import json
import nltk
from nltk.stem import PorterStemmer
#nltk.download('punkt')
from gensim.models import word2vec
import multiprocessing
from wordcloud import STOPWORDS
import re
from bs4 import BeautifulSoup
from calendar import monthrange
from datetime import datetime, timedelta
import string
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords

# Define a function to preprocess a tweet
def preprocess_tweet(tweet):
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove punctuation and special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Tokenize the tweet
    tokens = nltk.word_tokenize(tweet)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stem the words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    # Join the tokens back into a string
    tweet = ' '.join(tokens)
    return tweet


# Get Lexicons
with open('hatebase.txt', 'r') as file:
    hatebase = [line.strip() for line in file.readlines()]
#print(hatebase)

#Get Tweets
with open('hatespeech_labeledTweets.json') as json_file:
    data = json.load(json_file)

tweets = []
normal = data['normal']
hateful = data['hateful']
abusive = data['abusive']

tweets = hateful + abusive + normal
preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]
preprocessed_tweets = [nltk.word_tokenize(tweet.lower()) for tweet in tweets]
preprocessed_tweets = [[word for word in tweet if word not in string.punctuation] for tweet in preprocessed_tweets]

# Train a Word2Vec model on the preprocessed tweets
model = Word2Vec(preprocessed_tweets, min_count=1, vector_size=200)
model.save("my_word2vec_model")

# Find a similar word given a target word
#target_word = 'man'
#if target_word in model.wv.key_to_index:
#    similar_words = model.wv.most_similar(target_word, topn=10)
#    print(f"Words similar to {target_word}: {similar_words}")
#else:
#    print(f"{target_word} not found in the vocabulary")

similarWordDict = {}
for hateLexicon in hatebase:
	target_word = hateLexicon
	if target_word in model.wv.key_to_index:
		similar_words = model.wv.most_similar(target_word, topn=10)
		# Remove any similar words that appear in the hate_list
		similar_words = [(word, score) for (word, score) in similar_words if word not in hatebase]
		similarWordDict[target_word] = similar_words
	else:
		print(f"{target_word} not found in the vocabulary")


# Write the dictionary to a JSON file
with open('similarWordDict_2018.json', 'w') as outfile:
    json.dump(similarWordDict, outfile)

'''


#### SECOND ATTEMPT 25TH MARCH

'''
import gensim
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Read the hatebase lexicons from file
with open('hatebase.txt', 'r') as f:
    lexicons = [line.strip() for line in f]

#Get Tweets
with open('hatespeech_labeledTweets.json') as json_file:
    data = json.load(json_file)

tweets = []
normal = data['normal']
hateful = data['hateful']
abusive = data['abusive']

tweets = hateful + abusive + normal

# Tokenize the tweets and remove stop words
tweets_tokenized = [word_tokenize(tweet.lower()) for tweet in tweets]
stop_words = set(stopwords.words('english'))
tweets_filtered = [[word for word in tweet if word not in stop_words] for tweet in tweets_tokenized]

# Create a word embedding model using the tweets
model = gensim.models.Word2Vec(tweets_filtered, vector_size=100, window=5, min_count=5, workers=4)

# Find the top 10 most similar words for each word in the lexicons
similar_words = {}
for lexicon in lexicons:
    try:
        similar_words[lexicon] = [word for word, _ in model.wv.most_similar(lexicon, topn=10)]
    except KeyError:
        # Ignore words that are not in the word embedding model
        pass

# Save the top 10 most similar words for each lexicon to a JSON file
with open('similarWordDict_2018_v2.json', 'w') as f:
    json.dump(similar_words, f)

print("Output saved to similar_words.json")
'''

#### THIRD ATTEMPT 26th MARCH

import gensim
import gensim.downloader
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Read the hatebase lexicons from file
with open('hatebase.txt', 'r') as f:
    lexicons = [line.strip() for line in f]

#Get Tweets
with open('hatespeech_labeledTweets.json') as json_file:
    data = json.load(json_file)

tweets = []
normal = data['normal']
hateful = data['hateful']
abusive = data['abusive']

tweets = hateful + abusive + normal

# Load the pre-trained GloVe word embeddings
#glove_file = 'glove.6B.300d.txt'
#word_vectors = gensim.models.KeyedVectors.load_word2vec_format(glove_file, binary=False)

glove_vectors = gensim.downloader.load('glove-twitter-25')

# Tokenize the tweets and remove stop words
tweets_tokenized = [word_tokenize(tweet.lower()) for tweet in tweets]
stop_words = set(stopwords.words('english'))
tweets_filtered = [[word for word in tweet if word not in stop_words] for tweet in tweets_tokenized]

# Find the top 10 most similar words for each word in the lexicons
similar_words = {}
for lexicon in lexicons:
    try:
        similar_words[lexicon] = [word for word, _ in glove_vectors.most_similar(lexicon, topn=10)]
    except KeyError:
        # Ignore words that are not in the pre-trained word embeddings
        pass

# Save the top 10 most similar words for each lexicon to a JSON file
with open('similarWordDict_2018_v3.json', 'w') as f:
    json.dump(similar_words, f)

print("Output saved to similar_words.json")









