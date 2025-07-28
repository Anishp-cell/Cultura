import requests
import os
from dotenv import load_dotenv
load_dotenv()
QLOO_API_KEY = "AOlP87vLgupHXo0z_0w-LvjyMyToinHmC_xnE01hPqU"
url = "https://hackathon.api.qloo.com/v2/insights"
params ={
"filter.type": "urn:entity:tv_show",
"signal.interests.tags": "urn:tag:music:experimental",
"take":10,
}
headers = {"accept": "application/json",
           "x-api-key": QLOO_API_KEY}

response = requests.get(url, headers=headers, params=params)

print(response.text)
print(response.status_code)
print(response.json())

# we have access to only these entities:

# urn:entity:artist
# urn:entity:book
# urn:entity:brand
# urn:entity:destination
# urn:entity:movie
# urn:entity:person
# urn:entity:place
# urn:entity:podcast
# urn:entity:tv_show
# urn:entity:video_game

