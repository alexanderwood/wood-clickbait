import requests

import os

TWITTER_BEARER_TOKEN = os.environ['TWITTER_BEARER_TOKEN']

headers = {
    'Authorization': 'Bearer {}'.format(TWITTER_BEARER_TOKEN),
}


ids = ["1196042842398973952", "1195923245603667973"]

params = (
    ('ids', ",".join(ids)),
    ('expansions', 'author_id'),
    ('tweet.fields', 'created_at,public_metrics,entities'),
)

response = requests.get('https://api.twitter.com/2/tweets', headers=headers, params=params)

print(response.text)
