import os
import requests
from bs4 import BeautifulSoup as bs


class AuthMixin(object):
    def oauth2(self, bearer_token):
        if bearer_token is None:
            raise ValueError('Invalid Bearer Token')


class Hydrator(AuthMixin):
    def __init__(self, url=""):
        self.url = url
        self.auth = False


    def authenticate(self, bearer_token=None):
        pass


    def from_list(self, id_list, bearer_token=None, expansions=None,
                  tweet_fields=None):
        self.oauth2(bearer_token)
        header = {'Authorization': 'Bearer {}'.format(bearer_token)}

        endpoint_url = 'https://api.twitter.com/2/tweets'

        params = {'ids': ','.join(id_list)}
        if expansions:
            params['expansions'] = ','.join(expansions)
        if tweet_fields:
            params['tweet.fields'] = ','.join(tweet_fields)

        dat = requests.get(endpoint_url, params=params, headers=header)
        return dat



BEARER_TOKEN = os.environ['TWITTER_BEARER_TOKEN']
expansions = ['author_id']
tweet_fields=['created_at', 'public_metrics', 'entities']

H = Hydrator()
val = H.from_list(['1195923245603667973'], BEARER_TOKEN, expansions=expansions,
                  tweet_fields=tweet_fields)
print(val.text)
