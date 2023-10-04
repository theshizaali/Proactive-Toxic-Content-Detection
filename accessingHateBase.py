import requests
import json
import pandas as pd

# initialize variables
auth_url = "https://api.hatebase.org/4-2/authenticate"
vocab_url = "https://api.hatebase.org/4-2/get_vocabulary"
lang = "eng"
resp_format = "json"
token = ""
pages = 0
total_entries = 0

# initialize authentication payload
auth_payload = "api_key=mmKPRoEEPSCHTI34598dhtP"
headers = {
    'Content-Type': "application/x-www-form-urlencoded",
    'cache-control': "no-cache"
    }


# authenticate
auth_resp = requests.request("POST", auth_url, data=auth_payload, headers=headers)

# get the session token
token = auth_resp.json()["result"]["token"]

# quick check if it worked
print(auth_resp.text, token)

# initialize vocabulary payload
# first without any given page-number
vocab_payload = "token=" + token + "&format=" + resp_format + "&language=" + lang
voc_resp = requests.request("POST", vocab_url, data=vocab_payload, headers=headers)
print(voc_json = voc_resp.json())


# how many pages in total? and how many results= entries in the vocab?
pages = voc_json["number_of_pages"]
results = voc_json["number_of_results"]
# check pages & results
print(pages, results)

# create vocabulary df from first resultset
df_voc = pd.DataFrame(voc_json["result"])

# now get results of all remaining pages and append to df_voc
for page in range(2,pages+1):
    vocab_payload = "token=" + token + "&format=json&language=" + lang + "&page=" + str(page)
    voc_resp = requests.request("POST", vocab_url, data=vocab_payload, headers=headers)
    voc_json = voc_resp.json()
    df_voc = df_voc.append(voc_json["result"])

# reset df_voc index
df_voc.reset_index(drop=True, inplace=True)
print(df_voc.shape)

df_voc.to_csv("hatebase_vocab.csv")