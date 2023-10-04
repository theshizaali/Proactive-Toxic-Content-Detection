import csv
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Define the classification pipeline
classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer
)

# Define the input file and output file
input_file = "hatespeech_tweets.csv"
output_file = "classified_hatespeech_tweets.csv"

# Open the input file and read tweets
with open(input_file, "r") as f:
    reader = csv.reader(f)
    tweets = list(reader)

# Open the output file and write the classified tweets
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tweet", "hate_speech_probability"])
    for tweet in tweets:
        text = tweet[0]
        result = classifier(text)[0]
        label = result["label"]
        probability = result["score"]

        print(str(text) + str(probability))
        writer.writerow([text, probability])