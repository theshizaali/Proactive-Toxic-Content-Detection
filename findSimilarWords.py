from gensim.models import Word2Vec

# Train a Word2Vec model
sentences = [['she', 'bitch'], ['she', 'eyesroll', 'karen']]
model = Word2Vec(sentences, min_count=1, vector_size=100)

# Save the model to a file
model.save("my_word2vec_model")

# Load the model from the file
loaded_model = Word2Vec.load("my_word2vec_model")

# Use the loaded model
similar_words = loaded_model.wv.most_similar('bitch')
print(f"Words similar to 'bitch': {similar_words}")
