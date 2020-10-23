import requests
import os
from pathlib import Path
import json
from bs4 import BeautifulSoup


# Load the stored key/token
bearer_token = os.environ['TWITTER_BEARER_TOKEN']

def data_path():
    return Path(os.getcwd()) / "data"


def create_payload(id_list):
    payload = {
        'ids':id_list,
        'expansions':'author_id',
        'tweet.fields':'created_at,public_metrics,entities'
    }
    headers = {"authorization":"Bearer {}".format(bearer_token)}


def get_response(id_list):
    dat = requests.get("https://api.twitter.com/2/tweets",
        params=payload,
        headers=headers
    )


def load_ids(fpath=Path("webis-clickbait-17") / "raw-data" / "instances.jsonl"):
    # Load IDs from the provided file path, where the provided file is in the
    # working directory.
    p = data_path()
    tweet_ids = []
    with open(p / fpath) as f:
        line = f.readline()
        while line:
            dat = json.loads(line)
            tweet_ids.append(dat['id'])
            line = f.readline()

    return tweet_ids

tweet_ids = load_ids()
print(tweet_ids[0], len(tweet_ids))
