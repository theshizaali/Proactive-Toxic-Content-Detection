import csv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define a function to classify a single tweet
def classify_tweet(tweet):
    # Tokenize the tweet text
    encoded_input = tokenizer(tweet, padding=True, truncation=True, max_length=128, return_tensors='pt')

    # Make a prediction
    with torch.no_grad():
        output = model(**encoded_input)

    # Get the predicted label
    predicted_label = torch.softmax(output.logits, dim=1)[0][1].item()

    # Return the predicted label (probability of hate speech)
    return predicted_label

# Load the CSV file of tweets
with open('normal_tweets.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for row in reader:
        tweet_text = row[0]
        prob_hate_speech = classify_tweet(tweet_text)
        print("ok")
        if prob_hate_speech > 0.5:
            print('Tweet "{}" is hate speech (probability: {:.2f}%).'.format(tweet_text, prob_hate_speech * 100))
        #else:
            #print('Tweet "{}" is not hate speech (probability: {:.2f}%).'.format(tweet_text, prob_hate_speech * 100))


        

