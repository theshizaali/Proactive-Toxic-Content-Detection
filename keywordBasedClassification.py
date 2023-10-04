import json

with open("hatespeech_labeledTweets.json") as f:
    data = json.load(f)
#print(data)


with open("offensiveWords_hatebase.txt") as f:
    lines = f.readlines()

offensiveWords = [line.strip() for line in lines]
#print(offensiveWords)

normal = data['normal']
abusive = data['abusive']
hateful = data['hateful']


for tweet in normal:
	words = tweet.lower().split()
	for word in words:
		if word in offensiveWords:
			print(tweet)
			break



'''
def classify_text(text):
    categories = {
        "sports": ["basketball", "football", "soccer", "tennis"],
        "technology": ["computer", "mobile", "software"],
        "politics": ["government", "election", "politics"]
    }

    words = text.lower().split()

    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in words:
                return category

    return "Unknown"

text = "I love playing basketball and watching football games on weekends."
print(classify_text(text)) # Output: sports
'''