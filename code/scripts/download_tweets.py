import os
import sys
import csv

from pathlib import Path
from twython import Twython

from retrying import retry
import time

# Load the stored key/token
consumer_key = os.environ['TWITTER_CONSUMER_KEY']
BEARER_TOKEN = os.environ['TWITTER_BEARER_TOKEN']
consumer_secret = os.environ['TWITTER_CONSUMER_SECRET']

access_key = os.environ['TWITTER_ACCESS_TOKEN']
access_secret = os.environ['TWITTER_ACCESS_SECRET']

# Dictionary of news site API keys
SITE_KEY = {'nytimes':os.environ['NYT_ACCESS_KEY']}

def save_file(screen_name):
    return "tweet-data-{}.csv".format(screen_name)


def data_path():
    return Path(os.getcwd()).parents[0] / "data"


def authenticate():
    '''Create a Twython instance with your key and token'''
    return Twython(consumer_key, access_token=BEARER_TOKEN)

@retry(stop_max_attempt_number=1, wait_fixed=6000)
def query_user_timeline(twitter, screen_name, max_id):
    '''Query the Twitter API for @screen_name's tweets older than ID#'''
    return twitter.get_user_timeline(screen_name=screen_name,
                                       count=200,
                                       max_id=max_id,
                                       tweet_mode='extended')


def parse_response(tweet, screen_name):
    # Parse the tweet responses & save them to the csv file.
    with (data_path() / save_file(screen_name)).open(mode='a') as fout:
        writer = csv.writer(fout, delimiter=',')
        
        # If the tweet is neither a retweet nor a reply, add it to db.
        if ('retweeted_status' not in tweet.keys()) and (tweet['in_reply_to_status_id_str'] is None):
            full_text = tweet["full_text"]
            i = full_text.find("https://t.co/")
            hard_link = full_text[i:].lstrip()
            full_text = full_text[:i].rstrip()
            
            row = [tweet['user']['id_str'],
                   '"{}"'.format(tweet['user']['screen_name']),  # in case of comma
                   '"{}"'.format(tweet['user']['name']),
                   tweet['id_str'],
                   tweet['created_at'],
                   '"{}"'.format(full_text),
                   tweet['entities']['urls'][0]['url'],
                   tweet['entities']['urls'][0]['expanded_url'],
                   hard_link,
                   tweet['retweet_count'],
                   tweet['favorite_count']
                   ]
            writer.writerow(row)
        else:
            with (data_path() / "invalid.csv").open('a') as ferr:
                ferr.write('{},{}\n'.format(tweet['user'], tweet['id_str']))
        
        
def oldest_tweet_id(screen_name):
    '''
    Get the oldest tweet ID for a user, so we can continue
    querying (200 at at time maximum.)
    '''
    def tail(f):
        '''Get the last line of the file'''
        
        # Move pointer to the last 512 characters of file
        f.seek(f.seek(0,2) - 512, 0)
        
        # Grab that last row
        csvreader = csv.reader(f)
        last_row = ''
        for row in csvreader:
            last_row = row
            
        return last_row
        
        
    def find_default(f, screen_name):
        '''Get ID of last tweet on August 31'''
        csvreader = csv.reader(f)
        default_id = ''
        for row in csvreader:
            if row[0]==screen_name:
                default_id = row[1]
        return default_id
        
    # Path to data folder
    file_path = Path(os.getcwd()).parents[0] / "data"
    
    # If already started downloading tweets, then pickup where we left off.
    if os.path.exists(file_path / "tweet-data-{}.csv".format(screen_name)):
        with (file_path / "tweet-data-{}.csv".format(screen_name)).open("r") as fout:
            tweet_id = tail(fout)
            tweet_id = tweet_id[3]  # Tweet ID is the third column in CSV
    # Otherwise, get the starting ID for that username.
    else:
        with (file_path / "tweet-data-init.csv").open("r") as fout:
            tweet_id = find_default(fout, screen_name)
    
    return tweet_id
    

def scrape(twitter, screen_name, max_id=None):
    '''
    Scrape info on the last 200 tweets, starting with
    where the last loop left off.
    '''
    
    # Try and get the oldest tweet ID if it is not provided.
    if max_id is None:
        max_id = oldest_tweet_id(screen_name)

    # If the oldest tweet ID is found, then query the API.
    if max_id is not None:
        tweets = query_user_timeline(twitter, screen_name, max_id)
        
        for tweet in tweets:
            parse_response(tweet, screen_name)
            

def hydrate(ids):
    pass
'''
if __name__ == "__main__":
    screen_name = sys.argv[1]

    twitter = authenticate()
    
    for _ in range(10):
        scrape(twitter, screen_name)
        time.sleep(6)
        print(twitter.get_application_rate_limit_status()["resources"]["application"]["/application/rate_limit_status"]["remaining"])
'''




# the ID of the status
ID = "1217763838336077824"

