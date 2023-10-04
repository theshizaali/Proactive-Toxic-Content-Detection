'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")
model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/bert-base-uncased-hatexplain")

# Define the tweet text
text = "My father warned me to not get involved with currys"

# Tokenize the text and add special tokens
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Pass the input through the model to get the prediction
outputs = model(**inputs)
scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
labels = ['not-hate', 'hate']
predicted_label = labels[scores.argmax()]

print(f"The predicted label for the tweet '{text}' is {predicted_label}.")
'''
print("okay start")
'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig
import torch

# Define the hate speech lexicons
hate_lexicons = ["bigot", "chink", "coon", "dyke", "fag", "gook", "nigga", "raghead", "spic", "currys"]

# Load the pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
config = AutoConfig.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased", config=config)

# Add an additional input layer for hate speech lexicons
hate_lexicon_input = torch.randn(1, len(hate_lexicons))
model.resize_token_embeddings(config.vocab_size + len(hate_lexicons))

# Define the tweet text
text = "My father warned me to not get involved with currys"

# Tokenize the text and add special tokens
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# Enhance the input embeddings with hate speech lexicons
hate_lexicon_embeddings = model.embeddings.word_embeddings(torch.LongTensor([[tokenizer.vocab.get(w, tokenizer.vocab["[UNK]"]) for w in hate_lexicons]]))
inputs_embeds = inputs["input_ids"].new_zeros((1, inputs["input_ids"].size(-1), config.hidden_size))
inputs_embeds[:, :inputs["input_ids"].size(-1), :] = model.embeddings.word_embeddings(inputs["input_ids"])
inputs_embeds[:, -len(hate_lexicons):, :] = hate_lexicon_embeddings
inputs = {"inputs_embeds": inputs_embeds}

# Load the pre-trained BERT classification head
classification_head = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Pass the enhanced input through the classification head to get the prediction
outputs = classification_head(**inputs)
scores = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
labels = ['not-hate', 'hate']
predicted_label = labels[scores.argmax()]

print(f"The predicted label for the tweet '{text}' is {predicted_label}.")
'''
print("okay start")
'''
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load DistilBERT pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define tweet text
tweet = "my father warned me against currys"

# Tokenize tweet text
inputs = tokenizer.encode_plus(
    tweet,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# Predict label for tweet text
outputs = model(**inputs)
probas = torch.softmax(outputs[0], dim=1)
label = torch.argmax(probas)

if label == 1:
    print("The tweet is classified as hate speech")
else:
    print("The tweet is not classified as hate speech")
'''
'''
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch


hatebase_lexicons = set()
with open("hatebase.txt") as f:
    for line in f:
        hatebase_lexicons.add(line.strip())

hatebase_lexicons.add('currys')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')


def preprocess(text):
    # Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    # Check if any token is in the hatebase lexicons
    for token in tokens:
        if token.lower() in hatebase_lexicons:
            tokens.append("<HATEBASE>")
            break
    # Convert tokens to input IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Pad or truncate the input IDs
    input_ids = input_ids[:512] + [0] * (512 - len(input_ids))
    # Convert input IDs to tensor
    input_tensor = torch.tensor([input_ids])
    return input_tensor


import torch.nn.functional as F

def detect_hate_speech(text):
    input_tensor = preprocess(text)
    with torch.no_grad():
        logits = model(input_tensor)[0]
    probabilities = F.softmax(logits, dim=1).tolist()[0]
    # Label 1 is hate speech, label 0 is not hate speech
    return probabilities[1]


text = "I hate people who are currys."
probability = detect_hate_speech(text)
print(f"Hate speech probability: {probability:.2f}")
'''

from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# Load the BERT-base-uncased model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a list of hate speech lexicons
lexicons = ['snowflake', 'bigot', 'racist', 'homophobic', 'xenophobic', 'sexist', 'discrimination']

# Define a function to preprocess the tweet by tokenizing it and replacing the hate speech lexicons with a special token
def preprocess_tweet(tweet):
    for lexicon in lexicons:
        tweet = tweet.replace(lexicon, f"[{lexicon}]")
    encoded_tweet = tokenizer.encode_plus(
        tweet,
        max_length=128,
        add_special_tokens=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded_tweet

# Define a function to predict the label (hate speech or not hate speech)
def predict_label(tweet):
    encoded_tweet = preprocess_tweet(tweet)
    with torch.no_grad():
        logits = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])[0]
    probs = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()
    predicted_label = np.argmax(probs, axis=-1)
    if predicted_label == 1:
        print("The tweet is hate speech with a probability of {:.2f}%".format(probs[0][1]*100))
    else:
        print("The tweet is not hate speech with a probability of {:.2f}%".format(probs[0][0]*100))

# Test the model on a sample tweet
tweet = "Actions have consequences snowflake"
predict_label(tweet)



