from tdv2 import Hydrator
import os
import csv
from pathlib import Path
import json
import time

BEARER_TOKEN = os.environ['TWITTER_BEARER_TOKEN']
h = Hydrator(bearer_token=BEARER_TOKEN)

#'abc', 'bbc', 'breitbart', 'cbs', 'cnn', 'fox','msnbc', 'nbc',
handles = [ 'npr', 'wapo', 'wsj']

for handle in handles:
    print("Loading", handle)
    ################################################################################
    # Load the IDs.
    fname = "{}-2017.txt".format(handle)
    fpath = Path(os.getcwd()) / "data" / "news-outlets" / "tweet-ids" / fname

    '''
    tweet_ids = []
    with open(fpath) as f:
        line = f.readline()
        while line:
            dat = json.loads(line)
            tweet_ids.append(dat['id'])
            line = f.readline()

    '''
    tweet_ids = []
    with open(fpath) as f:
        tweet_ids = f.readlines()
    tweet_ids = [tweet_id.rstrip() for tweet_id in tweet_ids]

    print("Number of tweets:", len(tweet_ids))

    ################################################################################
    # Load the (already loaded) IDs.

    ################################################################################
    # Hydrate the tweets and save them.
    master_parameters = {
        'expansions': ['author_id'],
        'tweet.fields': ['created_at', 'public_metrics', 'entities']
    }

    # File for saving.
    save_file = Path(os.getcwd()) / "data" / "news-outlets" / fname.replace(".txt", ".csv")

    # Main hydrator function.
    step=100
    for i in range(0, len(tweet_ids), step):

        # The next batch.
        j = min(i + step, len(tweet_ids))
        print(i, j)

        ############################################################################
        # Hydrate the tweets.
        payload = h.hydrate_list(ids=tweet_ids[i:j], args=master_parameters)
        payload = json.loads(payload.text)

        ############################################################################
        # Make a hash map of the user IDs in the payload
        name_info = {}
        for user in payload['includes']['users']:
            name_info[user['id']] = (user['name'], user['username'])

        ############################################################################
        # Get each row, write to a CSV.
        for tweet in payload['data']:
            row = []
            row.append(tweet['id']),
            row.append(tweet['author_id'])
            row.append(name_info[tweet['author_id']][0]),
            row.append(name_info[tweet['author_id']][1]),
            row.append(tweet['created_at'])
            row.append(tweet['public_metrics']['retweet_count'])
            row.append(tweet['public_metrics']['reply_count'])
            row.append(tweet['public_metrics']['like_count'])
            row.append(tweet['public_metrics']['quote_count'])
            row.append(tweet['text'])
            if 'entities' in tweet.keys():
                if 'unwound_url' in tweet['entities']['urls'][0].keys():
                    row.append(tweet['entities']['urls'][0]['unwound_url'])
                else:
                    row.append(tweet['entities']['urls'][0]['url'])
            else:
                row.append('')


            with open(save_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(row)

        time.sleep(9)  # Wait between api calls for time limit.